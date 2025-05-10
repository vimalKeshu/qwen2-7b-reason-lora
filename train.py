#!/usr/bin/env python3
"""
LoRA + DeepSpeed training for Qwen2-7B on ConfinityChatMLv1 dataset for logical reasoning.
Includes evaluation on GSM8K and MMLU benchmarks.
"""
import argparse
import os
import json
import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from transformers.trainer_utils import is_main_process

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2-7B")
    parser.add_argument("--output_dir", required=True, help="Directory to save the model")
    parser.add_argument("--deepspeed", required=True, help="Path to ds_config_zero2.json")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training")
    parser.add_argument("--evaluate_after_training", action="store_true", 
                       help="Run GSM8K and MMLU evaluation after training")
    
    # LoRA specific arguments
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Evaluation specific arguments
    parser.add_argument("--evaluation_batch_size", type=int, default=8, 
                       help="Batch size for evaluation")
    
    return parser.parse_args()

def prepare_dataset(tokenizer, max_seq_length):
    """
    Prepare ConfinityChatMLv1 dataset for fine-tuning.
    """
    logger.info("Loading ConfinityChatMLv1 dataset...")
    raw_dataset = load_dataset("vmal/ConfinityChatMLv1")
    
    # Split for train/validation
    split_dataset = raw_dataset["train"].train_test_split(test_size=0.005, seed=42)
    train_dataset, val_dataset = split_dataset["train"], split_dataset["test"]
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # The dataset is already in ChatML format
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )
    
    logger.info("Tokenizing train dataset...")
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenizing train dataset",
    )
    
    logger.info("Tokenizing validation dataset...")
    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenizing validation dataset",
    )
    
    return train_tokenized, val_tokenized

def evaluate_model(args):
    """
    Evaluate the model on GSM8K and MMLU benchmarks
    """
    from lm_eval import evaluator
    logger.info("Starting evaluation on GSM8K and MMLU benchmarks...")
    
    # Define tasks for evaluation
    tasks = ["mmlu", "gsm8k"]
    
    # GSM8K uses 5-shot examples for best results
    tasks_config = {
        "mmlu": {"num_fewshot": 5},
        "gsm8k": {"num_fewshot": 8}
    }
    
    # Evaluation results for both tasks
    all_results = {}
    
    for task in tasks:
        logger.info(f"Evaluating model on {task}...")
        num_fewshot = tasks_config[task]["num_fewshot"]
        
        # Run evaluation using lm-evaluation-harness
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={args.output_dir},trust_remote_code=True",
            tasks=[task],
            num_fewshot=num_fewshot,
            batch_size=args.evaluation_batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        # Log results
        logger.info(f"Results for {task}:")
        logger.info(results["results"])
        
        # Store results
        all_results[task] = results["results"]
    
    # Save evaluation results to a file
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_path}")
    return all_results

def main():
    args = parse_args()
    
    # Set up distributed training if needed
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        
    logger.info("🚀 Starting training...")
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"DeepSpeed config: {args.deepspeed}")
    
    # Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    # Qwen2 has no pad token, so use eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    train_dataset, val_dataset = prepare_dataset(tokenizer, args.max_seq_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Base model
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    
    # Move model to appropriate device
    device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1 else "cuda")
    base_model.to(device)
    
    # Enable Flash Attention 2 after moving to device
    logger.info("Enabling Flash Attention 2...")
    for module in base_model.modules():
        if hasattr(module, "attn_implementation"):
            module.attn_implementation = "flash_attention_2"
    
    # Enable gradient checkpointing
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()
    
    # Configure LoRA
    logger.info("Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to model
    model = get_peft_model(base_model, lora_config)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,    
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        deepspeed=args.deepspeed,
        report_to="none",
        ddp_find_unused_parameters=False,
        warmup_steps=args.warmup_steps,
        fp16=False,  # Use bf16 instead for better stability
        remove_unused_columns=False,  # Important for some model architectures       
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save model and tokenizer
    if is_main_process(training_args.local_rank):
        logger.info("Saving model...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluate on benchmarks if requested
    if args.evaluate_after_training and is_main_process(training_args.local_rank):
        logger.info("Evaluating on benchmarks...")
        evaluate_model(args)

if __name__ == "__main__":
    main()