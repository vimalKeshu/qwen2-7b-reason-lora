#!/bin/bash

echo "Install git-lfs.."
apt update && apt install git-lfs

# Install required packages
echo "Installing required packages..."
# pip install -q torch>=2.1.0
# pip install flash-attn --no-build-isolation
pip install -r requirements.txt

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
export NCCL_DEBUG_SUBSYS=TOPO

# Define CUDA visible devices (adjust as needed)
export CUDA_VISIBLE_DEVICES=0,1  # Use 2 GPUs

# Execute the training script
echo "Starting fine-tuning process..."
nohup \
deepspeed --num_gpus=2 train.py \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --deepspeed $DEEPSPEED_CONFIG \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --logging_steps $LOGGING_STEPS  > 1.log 2>&1 &
    # --evaluate_after_training
   