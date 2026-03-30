"""
Supplementary Experiment #1 — Random Noise Baseline
=====================================================
Purpose: Replace the RND exploration reward with Gaussian noise N(0, sigma^2),
         added to incorrect-trajectory advantages (all other logic unchanged).
         Compares against the main method on Countdown-4 to verify that the
         prediction-error mechanism (not merely "perturbing incorrect trajectories")
         is the source of performance gains.

Design principles:
- Only change: RND -> Gaussian noise in the exploration reward signal.
  All other logic (error-conditioned allocation, advantage-preserving integration,
  scales attenuation) is identical to main_iMENTOR.py.
- No existing files are modified (main_iMENTOR.py / main_grpo.py /
  core_algos.py / ray_trainer.py remain untouched).
- Noise parameters are controlled via the random_noise yaml config block.

Usage:
    bash scripts/random_noise_countdown-4_3b.sh
"""

from verl import DataProto
import torch
import ray
from verl.utils.reward_score import gsm8k, countdown
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _select_rm_score_fn(data_source):
    """Select the reward scoring function based on data source."""
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    else:
        raise NotImplementedError(f"Unsupported data_source: {data_source}")


# ---------------------------------------------------------------------------
# GaussianNoiseActor — replaces RNDActor from main_iMENTOR.py
# ---------------------------------------------------------------------------

@ray.remote(num_gpus=1)
class GaussianNoiseActor:
    """
    Ray remote actor occupying 1 dedicated GPU (same resource interface as RNDActor).

    Core logic: no network is trained. For each sequence in the batch, sample
    a scalar from N(0, sigma^2) and apply min-max normalization to [0, 0.5],
    matching the normalization used in RNDReward.forward.

    Args:
        sigma (float): Standard deviation of the Gaussian noise.
                       Corresponds to random_noise.sigma in the yaml config.
    """

    def __init__(self, sigma: float):
        self.device = torch.device("cuda:0")
        self.sigma = sigma

    def sample_one_batch(self, batch_size: int) -> torch.Tensor:
        """
        Sample a scalar Gaussian noise for each sequence in the batch,
        then apply min-max normalization to [0, 0.5].

        Normalization formula (identical to RNDReward.forward x1 computation):
            normalized = 0.5 * (x - x_min) / (x_max - x_min)

        Args:
            batch_size (int): Number of sequences in the current batch.

        Returns:
            noise_reward (torch.Tensor): shape (batch_size, 1), CPU tensor.
        """
        # Sample raw Gaussian noise, shape: (batch_size, 1)
        raw_noise = torch.randn(batch_size, 1, device=self.device) * self.sigma

        # Min-max normalization to [0, 0.5]
        x_min = raw_noise.min()
        x_max = raw_noise.max()
        if (x_max - x_min).abs() < 1e-8:
            # Edge case: all values identical, normalize to 0
            normalized = torch.zeros_like(raw_noise)
        else:
            normalized = 0.5 * (raw_noise - x_min) / (x_max - x_min)

        return normalized.detach().cpu()


# ---------------------------------------------------------------------------
# RandomNoiseRewardManager — replaces RewardManager from main_iMENTOR.py
# ---------------------------------------------------------------------------

class RandomNoiseRewardManager:
    """
    Reward Manager for the Random Noise Baseline experiment.

    The only difference from the main method's RewardManager:
        RNDActor.train_one_batch(rnd_inputs)
            -> GaussianNoiseActor.sample_one_batch(batch_size)

    All other logic is identical: error-conditioned allocation, scales
    attenuation, and advantage-preserving integration.

    Args:
        tokenizer: HuggingFace tokenizer.
        num_examine (int): Number of samples to print per data source for debugging.
        sigma (float): Gaussian noise standard deviation.
        scales (list): [alpha_scale, gamma_curr, gamma_step]
                       alpha_scale -- max exploration reward intensity (alpha in the paper)
                       gamma_curr  -- current decay denominator (gamma, incremented each step)
                       gamma_step  -- increment per step (fixed at 1)
    """

    def __init__(self, tokenizer, num_examine: int, sigma: float, scales: list) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        # Initialize GaussianNoiseActor on 1 dedicated GPU
        self.noise_actor = GaussianNoiseActor.remote(sigma)
        self.scales = scales  # [alpha_scale, gamma_curr, gamma_step]

    def __call__(self, data: DataProto):
        """
        Compute outcome reward and random noise exploration reward.

        Returns:
            reward_tensor (torch.Tensor): outcome-based reward, shape (batch, response_len)
            noise_reward_tensor (torch.Tensor): Gaussian noise exploration reward,
                                                shape (batch, response_len)
        """
        # If rm_scores are already computed, return directly
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        batch_size = len(data)

        # Initialize reward tensors
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        noise_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # ----------------------------------------------------------------
        # Step 1: Sample noise rewards via GaussianNoiseActor (async Ray call)
        # ----------------------------------------------------------------
        noise_1 = ray.get(self.noise_actor.sample_one_batch.remote(batch_size))
        # noise_1: shape (batch_size, 1)

        already_print_data_sources = {}

        # ----------------------------------------------------------------
        # Step 2: Compute outcome reward per sequence;
        #         apply error-conditioned noise reward allocation.
        # ----------------------------------------------------------------
        for i in range(batch_size):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode full sequence (prompt + response)
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # Compute outcome-based reward score
            score = compute_score_fn(
                error_score=0.0,
                format_score=0.1,
                solution_str=sequences_str,
                ground_truth=ground_truth
            )
            # Write outcome reward at the EOS token position
            reward_tensor[i, valid_response_length - 1] = score

            # ----------------------------------------------------------------
            # Error-conditioned reward allocation (identical to main method):
            # Only assign exploration reward to incorrect responses (score < 1.0)
            # ----------------------------------------------------------------
            if score < 1.0:
                # scales[0] = alpha_scale (max intensity)
                # scales[1] = gamma_curr  (decay denominator)
                noise_reward_tensor[i, valid_response_length - 1] = (
                    noise_1[i] * self.scales[0] / self.scales[1]
                )

            # Debug: print a few samples per data source
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

        print(sequences_str)

        # ----------------------------------------------------------------
        # Step 3: Update gamma_curr — attenuation schedule
        # scales = [alpha_scale, gamma_curr, gamma_step]
        # gamma_curr += gamma_step each step, so exploration decays over training
        # ----------------------------------------------------------------
        self.scales[1] += self.scales[2]

        return reward_tensor, noise_reward_tensor


# ---------------------------------------------------------------------------
# ValRewardManager — identical to main_iMENTOR.py, for validation only
# ---------------------------------------------------------------------------

class ValRewardManager:
    """Validation Reward Manager. Uses outcome reward only (no exploration reward)."""

    def __init__(self, tokenizer, num_examine: int) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # Validation: no format reward (format_score=0.0), accuracy only
            score = compute_score_fn(
                error_score=0.0,
                format_score=0.0,
                solution_str=sequences_str,
                ground_truth=ground_truth
            )
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    """Random Noise Baseline training entry point.

    Uses the same ppo_trainer.yaml config as main_iMENTOR.py, with an
    additional random_noise config block:
        random_noise:
          sigma: 1.0          # Gaussian noise standard deviation
          scales: [40.0, 40.0, 1.0]  # Same attenuation parameters as main method
    """
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from omegaconf import OmegaConf
    from pprint import pprint

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # Build worker classes (identical to main_iMENTOR.py)
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # ----------------------------------------------------------------
    # Key difference: use RandomNoiseRewardManager instead of RewardManager.
    # sigma and scales are read from the random_noise config block.
    # ----------------------------------------------------------------
    reward_fn = RandomNoiseRewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        sigma=config.random_noise.sigma,
        scales=list(config.random_noise.scales)
    )

    val_reward_fn = ValRewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping
    )

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
