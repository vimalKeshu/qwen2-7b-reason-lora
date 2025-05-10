#!/bin/bash

# Define parameters
export HF_HOME=/workspace/hf
export MODEL_NAME="Qwen/Qwen2-7B"
export OUTPUT_DIR="/workspace/qwen2-7b-logical-reasoning"
export DEEPSPEED_CONFIG="/workspace/ds_config_zero2.json"
export TORCH_EXTENSIONS_DIR="/workspace/torch_extensions"
export NUM_EPOCHS=3
export DEVICE_BATCH_SIZE=1
export GRAD_ACCUM=16
export LEARNING_RATE=2e-4
export MAX_SEQ_LENGTH=2048
export SAVE_STEPS=1000
export SAVE_TOTAL_LIMIT=3
export LOGGING_STEPS=100
export EVAL_STEPS=1000  # Run evaluation every 1000 steps
export LM_EVAL_CACHE=./lm_eval
export NCCL_DEBUG=INFO

# Define CUDA visible devices (adjust as needed)
export CUDA_VISIBLE_DEVICES=0,1  # Use 2 GPUs


# Run standalone evaluation (optional)
echo "Running standalone evaluation on GSM8K and MMLU..."
python -c "
from train import evaluate_model
import argparse

class Args:
    def __init__(self):
        self.output_dir = '$OUTPUT_DIR'
        self.evaluation_batch_size = 8

evaluate_model(Args())
"

echo "Evaluation complete!"