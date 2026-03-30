#!/usr/bin/env bash
# =============================================================================
# Supplementary Experiment #1 — Random Noise Baseline
# =============================================================================
# Purpose: Replace the RND exploration signal with Gaussian noise N(0, sigma^2),
#          keeping all other logic identical to the main method.
#          Run on Countdown-4 to compare against the main method and GRPO baseline,
#          verifying that prediction-error signal (not merely perturbing incorrect
#          trajectories) is the source of performance gain.
#
# Usage: bash scripts/random_noise_countdown-4_3b.sh
#
# Differences from iMENTOR_countdown-4_3b.sh:
#   1. Calls main_random_noise instead of main_iMENTOR
#   2. Uses random_noise yaml config block (sigma + scales) instead of imentor block
#      - sigma=1.0: Gaussian noise standard deviation
#      - scales=[40.0,40.0,1.0]: same attenuation parameters as the main method
#
# Resources: Same as main method — 4 GPUs + 1 extra GPU for GaussianNoiseActor
# =============================================================================

set -x
export DIR=./
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_random_noise \
    algorithm.adv_estimator=grpo \
    data.train_files=$DIR/data/countdown-4/train.parquet \
    data.val_files=$DIR/data/countdown-4/test.parquet \
    data.train_batch_size=512 \
    data.val_batch_size=512 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$DIR/models/Qwen2.5-3B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='countdown' \
    trainer.experiment_name='4-grpo-random-noise-3b' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=4 \
    random_noise.sigma=1.0 \
    random_noise.scales=[40.0,40.0,1.0] \
    2>&1 | tee random-noise-countdown-4-3b.log
