#!/usr/bin/env bash
# =============================================================================
# Supplementary Experiment #2 — Base Model Evaluation
# =============================================================================
# Purpose: Evaluate the base LLM (without any RL fine-tuning) on
#          GSM8K / Countdown-34 / Countdown-4, to report baseline accuracy
#          and disentangle the contribution of RL training from the main method.
#
# Usage: bash scripts/eval_base_model.sh
#
# Output:
#   - Terminal output: per-dataset pass@1 accuracy (ready to paste into paper table)
#   - JSON results saved to results/base_model_eval.json
#
# Prerequisites:
#   Run data preprocessing first:
#     python examples/data_preprocess/gsm8k.py --local_dir ./data/gsm8k
#     python examples/data_preprocess/countdown.py --local_dir ./data/countdown-34
#     python examples/data_preprocess/countdown-4.py --local_dir ./data/countdown-4
#
# Resource requirements:
#   - Only 1 GPU needed (no Ray distributed setup, pure vLLM inference)
#   - Estimated runtime: 30-60 minutes depending on GPU speed
# =============================================================================

set -x
export DIR=./

mkdir -p results

python scripts/eval_base_model.py \
    --model_path $DIR/models/Qwen2.5-3B \
    --data_files \
        gsm8k:$DIR/data/gsm8k/test.parquet \
        countdown-34:$DIR/data/countdown-34/test.parquet \
        countdown-4:$DIR/data/countdown-4/test.parquet \
    --max_tokens 1024 \
    --temperature 1.0 \
    --num_samples 1 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --output_file results/base_model_eval.json \
    2>&1 | tee results/base_model_eval.log
