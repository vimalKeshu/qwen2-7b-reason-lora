#!/usr/bin/env python3
"""
A callback for Hugging Face Trainer that evaluates the model on benchmarks
like GSM8K and MMLU at specific checkpoints during training.
"""
import os
import json
import time
import logging
from transformers import TrainerCallback, TrainerState, TrainerControl
from lm_eval import evaluator

logger = logging.getLogger(__name__)

class HarnessEvalCallback(TrainerCallback):
    """
    Trainer callback that runs benchmark evaluations at specified checkpoints.
    """
    def __init__(
        self,
        output_dir,
        model_name_or_path,
        tokenizer,
        eval_steps=None,
        eval_tasks=["mmlu", "gsm8k"],
        task_configs=None,
        batch_size=8,
    ):
        """
        Initialize the callback.
        
        Args:
            output_dir: Directory where evaluation results will be saved
            model_name_or_path: Path to model being trained
            tokenizer: Tokenizer instance
            eval_steps: At which steps to run evaluations, set to None for only end of training
            eval_tasks: List of tasks to evaluate (e.g., ["mmlu", "gsm8k"])
            task_configs: Dictionary of task-specific configurations
            batch_size: Batch size for evaluation
        """
        self.output_dir = output_dir
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.eval_tasks = eval_tasks
        self.batch_size = batch_size
        
        # Default task configs
        self.task_configs = {
            "mmlu": {"num_fewshot": 5},
            "gsm8k": {"num_fewshot": 8}
        }
        
        # Override with user provided configs
        if task_configs:
            for task, config in task_configs.items():
                if task in self.task_configs:
                    self.task_configs[task].update(config)
                else:
                    self.task_configs[task] = config
        
        # Create results directory
        os.makedirs(os.path.join(output_dir, "benchmark_results"), exist_ok=True)
        
        logger.info(f"HarnessEvalCallback initialized with tasks: {self.eval_tasks}")
        logger.info(f"Evaluation steps: {self.eval_steps}")
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Run evaluation at specified steps.
        """
        if self.eval_steps is None:
            return
        
        if state.global_step % self.eval_steps == 0:
            logger.info(f"Running benchmark evaluation at step {state.global_step}")
            
            # Save current checkpoint to a temporary directory
            temp_output_dir = os.path.join(self.output_dir, f"temp_step_{state.global_step}")
            os.makedirs(temp_output_dir, exist_ok=True)
            
            # Get model from kwargs
            model = kwargs.get("model", None)
            if model is None:
                logger.warning("Model not found in kwargs, skipping evaluation")
                return
            
            # Save model and tokenizer
            model.save_pretrained(temp_output_dir)
            self.tokenizer.save_pretrained(temp_output_dir)
            
            # Run evaluation
            all_results = self._run_evaluation(temp_output_dir)
            
            # Save results
            results_file = os.path.join(
                self.output_dir, 
                "benchmark_results", 
                f"results_step_{state.global_step}.json"
            )
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)
            
            logger.info(f"Benchmark results at step {state.global_step} saved to {results_file}")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Run evaluation at the end of training.
        """
        logger.info("Running final benchmark evaluation")
        
        # Run evaluation on the final model
        all_results = self._run_evaluation(self.output_dir)
        
        # Save results
        results_file = os.path.join(
            self.output_dir, 
            "benchmark_results", 
            "results_final.json"
        )
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Final benchmark results saved to {results_file}")
    
    def _run_evaluation(self, model_path):
        """
        Run evaluations on all specified tasks.
        
        Args:
            model_path: Path to the model to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        all_results = {}
        
        for task in self.eval_tasks:
            logger.info(f"Evaluating model on {task}...")
            
            # Get task configuration
            task_config = self.task_configs.get(task, {"num_fewshot": 0})
            num_fewshot = task_config.get("num_fewshot", 0)
            
            # Start timing
            start_time = time.time()
            
            try:
                # Run evaluation
                results = evaluator.simple_evaluate(
                    model="hf",
                    model_args=f"pretrained={model_path},trust_remote_code=True",
                    tasks=[task],
                    num_fewshot=num_fewshot,
                    batch_size=self.batch_size,
                )
                
                # Log results
                logger.info(f"Results for {task}:")
                logger.info(results["results"])
                
                # Store results
                all_results[task] = {
                    "results": results["results"],
                    "config": task_config,
                    "time_taken": time.time() - start_time
                }
                
            except Exception as e:
                logger.error(f"Error evaluating model on {task}: {e}")
                all_results[task] = {
                    "error": str(e),
                    "time_taken": time.time() - start_time
                }
        
        return all_results