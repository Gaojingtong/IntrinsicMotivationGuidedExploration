# IMAGINE codes

## Installation

```
conda create -n zero python=3.9

# activate it: conda activate zero

# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1

# ray
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation

# quality of life
pip install wandb IPython matplotlib
```

## An example for running

```
# First, please log in to your wandb with "wandb login"

# Then download the base model into ./models/
# if you haven't activated your environment: conda activate zero

# We usually store our dataset at ./data/
python ./examples/data_preprocess/countdown-4.py --local_dir {path_to_your_dataset}

# An example for running with grpo, you should first check the variable settings in it before running
bash scripts/iMENTOR_countdown-4_3b.sh
```

## Note
Due to the update of our code, we moved the reward network that was originally trained on CPU in the paper to the GPU. 

However, due to the author's limited understanding of ray and distributed computing. At present, we are unable to incorporate the training of the reward network into the existing GPU resourse pool.

Therefore, this project will additionally use 1 GPU to train the reward network separately (if the GPU number written in the scripts is 4, then actually a maximum of 4 + 1 = 5 GPUs will be used in training). This action will significantly enhance the overall training efficiency. 

## Acknowledge

- The project is conducted on the [TinyZero](https://github.com/Jiayi-Pan/TinyZero) base environment with reinforcement learning tool [verl](https://github.com/volcengine/verl).

- We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5)

