# Fine-tuning Qwen2-7B for Logical Reasoning

This project implements fine-tuning of the Qwen2-7B model using LoRA and DeepSpeed ZeRO stage 2 to create a model specialized in logical reasoning. The model, [qwen2-7b-logical-reasoning](https://huggingface.co/vmal/qwen2-7b-logical-reasoning), is trained on the [ConfinityChatMLv1](https://huggingface.co/datasets/vmal/ConfinityChatMLv1) dataset and evaluated on GSM8K and MMLU benchmarks.

## Overview

The implementation combines several advanced techniques:

1. **LoRA (Low-Rank Adaptation)**: Enables efficient fine-tuning of large language models with significantly reduced memory requirements
2. **DeepSpeed ZeRO Stage 2**: Optimizes memory usage through optimizer state partitioning and gradient partitioning
3. **Flash Attention 2**: Accelerates attention computation for faster training
4. **ConfinityChatMLv1 dataset**: A high-quality dataset for logical reasoning, already formatted in ChatML format
5. **Benchmark evaluation**: Evaluates the model on GSM8K and MMLU benchmarks

## Requirements

- Python 3.8+
- PyTorch 2.2.0+
- Transformers 4.37.0+
- PEFT 0.9.0+
- DeepSpeed 0.13.0+
- lm-eval 0.4.0+
- 2 GPUs with >=24GB VRAM (for distributed training)

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Files Description

- `train.py`: Main training script
- `ds_config_zero2.json`: DeepSpeed ZeRO Stage 2 configuration file
- `harness_eval_callback.py`: Callback for evaluation on benchmarks during training
- `run_train.sh`: Bash script to execute the training process
- `run_eval.sh`: Bash script to execute the eval process

## Usage

1. Make sure the scripts are executable:
   ```bash
   chmod +x run_train.sh
   chmod +x run_eval.sh
   ```

2. Run the training script:
   ```bash
   ./run_train.sh
   ```
3. Run the eval script:
   ```bash
   ./run_eval.sh
   ```   

## Training Configuration

The default configuration uses:
- LoRA with rank 64 and alpha 128
- BF16 mixed precision
- Batch size of 1 per GPU with gradient accumulation of 16
- Learning rate of 2e-4 with warmup
- Sequence length of 2048
- DeepSpeed ZeRO Stage 2 with CPU offloading

## Evaluation

The model is evaluated on:

1. **GSM8K**: Grade School Math Word Problems (8-shot evaluation)
2. **MMLU**: Massive Multitask Language Understanding (5-shot evaluation)

Evaluation results are saved to the output directory in JSON format.

## Advanced: Customizing the Training

You can customize various aspects of the training:

1. **Modify LoRA parameters**: Change `--lora_r`, `--lora_alpha`, and `--lora_dropout`
2. **Change DeepSpeed configuration**: Edit the `ds_config_zero2.json` file
3. **Adjust sequence length**: Change `--max_seq_length` based on your GPU memory
4. **Change evaluation frequency**: Modify `--eval_steps` in the run script

## Tips for Better Performance

1. **Increase training epochs**: For better performance, consider increasing `--num_train_epochs` to 5-10
2. **Adjust learning rate**: A learning rate between 1e-5 and 5e-4 usually works well
3. **Experiment with LoRA rank**: Increasing rank (64-256) can improve performance at the cost of more memory
4. **GPU memory optimization**: If facing OOM issues, try reducing batch size or sequence length

## License

This project is open-source and follows the same license as the base model and dataset.

## Acknowledgements

This implementation draws from techniques in the following papers and projects:
- LoRA: Efficient Finetuning
- ZeRO: Memory Optimizations Toward Training Billion Parameter Models
- DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters
- Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness