#!/usr/bin/env python
"""
Shared orchestration framework for OpenML evaluation scripts.

This module provides reusable functionality for:
1. Command construction for LLM baseline evaluations
2. Results aggregation across tasks and splits
3. Metadata validation and filtering
4. Task processing with error handling

Used by:
- examples/tabular/openml_cc18/run_openml_cc18_llm_baselines_tabular.py
- examples/tabular/openml_regression_2025/run_openml_regression_2025_llm_baselines_tabular.py
"""

import os
import json
import logging
import subprocess
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class OpenMLEvaluationOrchestrator:
    """
    Orchestrates LLM baseline evaluations on OpenML tasks with shared functionality.
    """
    
    def __init__(self, args, task_type: str = "classification"):
        """
        Initialize the orchestrator.
        
        Args:
            args: Command line arguments
            task_type: Type of tasks being evaluated ("classification" or "regression")
        """
        self.args = args
        self.task_type = task_type
        self.logger = logging.getLogger(f"{__name__}.{task_type}")
        
    def build_evaluation_command(self, task, split_idx: int) -> List[str]:
        """
        Build the command for evaluating LLM baselines on a specific task and split.
        
        Args:
            task: OpenML task object
            split_idx: Index of the split to use
            
        Returns:
            List of command arguments
        """
        task_id = task.task_id
        dataset_name = task.get_dataset().name
        
        # Create output directory
        eval_output_dir = os.path.join(
            self.args.output_dir, 
            f"task_{task_id}", 
            f"split_{split_idx}", 
            "llm_baselines"
        )
        os.makedirs(eval_output_dir, exist_ok=True)
        
        # Check if results already exist and skip if not forcing rerun
        results_file = os.path.join(eval_output_dir, "aggregated_results.json")
        if os.path.exists(results_file) and not getattr(self.args, 'force_rerun', False):
            self.logger.info(f"Results already exist for task {task_id}, split {split_idx+1}. Skipping.")
            return None
        
        # Generate version tag based on date for W&B project
        today = datetime.now()
        version_by_date = f"v{today.strftime('%Y%m%d')}"
        wandb_project = f"{self.args.wandb_project}-{version_by_date}"
        
        # Build evaluation command using the LLM baselines script
        eval_script = os.path.join(self.args.clam_repo_path, "examples", "tabular", "evaluate_llm_baselines_tabular.py")
        
        cmd = [
            "python", eval_script,
            "--task_ids", str(task_id),
            "--output_dir", eval_output_dir,
            "--models"
        ]
        
        # Add models as separate arguments (space-separated, not comma-separated)
        models_list = [model.strip() for model in self.args.models.split(',')]
        cmd.extend(models_list)
        
        # Add core arguments
        cmd.extend([
            "--num_few_shot_examples", str(self.args.num_few_shot_examples),
            "--seed", str(self.args.seed + split_idx),  # Vary seed for different splits
            "--device", self.args.device,
        ])
        
        # Add task-type specific arguments
        if self.task_type == "regression":
            cmd.append("--preserve_regression")  # Ensure regression tasks remain continuous
        
        # Add optional parameters
        if self.args.max_test_samples:
            cmd.extend(["--max_test_samples", str(self.args.max_test_samples)])
        
        # Add feature selection parameter
        cmd.extend(["--feature_selection_threshold", str(self.args.feature_selection_threshold)])
        
        # Add backend parameters
        cmd.extend([
            "--backend", self.args.backend,
            "--tensor_parallel_size", str(self.args.tensor_parallel_size),
            "--gpu_memory_utilization", str(self.args.gpu_memory_utilization),
        ])
        
        # Add CLAM-T-SNe specific parameters
        cmd.extend([
            "--vlm_model_id", self.args.vlm_model_id,
            "--embedding_size", str(self.args.embedding_size),
            "--tsne_perplexity", str(self.args.tsne_perplexity),
            "--tsne_max_iter", str(getattr(self.args, 'tsne_max_iter', getattr(self.args, 'tsne_n_iter', 1000))),
            "--max_tabpfn_samples", str(self.args.max_tabpfn_samples),
            "--nn_k", str(self.args.nn_k),
            "--max_vlm_image_size", str(self.args.max_vlm_image_size),
            "--image_dpi", str(self.args.image_dpi),
        ])
        
        # Add CLAM-T-SNe boolean flags
        if self.args.use_3d:
            cmd.append("--use_3d")
        if self.args.use_knn_connections:
            cmd.append("--use_knn_connections")
        if self.args.force_rgb_mode:
            cmd.append("--force_rgb_mode")
        else:
            cmd.append("--no-force_rgb_mode")
        if self.args.save_sample_visualizations:
            cmd.append("--save_sample_visualizations")
        else:
            cmd.append("--no-save_sample_visualizations")
        if self.args.use_semantic_names:
            cmd.append("--use_semantic_names")
        
        # Add visualization save cadence
        cmd.extend(["--visualization_save_cadence", str(self.args.visualization_save_cadence)])
        
        # Add custom viewing angles if specified
        if self.args.viewing_angles:
            cmd.extend(["--viewing_angles", self.args.viewing_angles])
        
        # Add model-specific arguments
        cmd.extend([
            "--tabllm_model", self.args.tabllm_model,
            "--tabula_model", self.args.tabula_model,
            "--jolt_model", self.args.jolt_model,
        ])
        
        # Add API model arguments if specified
        if hasattr(self.args, 'openai_model') and self.args.openai_model:
            cmd.extend(["--openai_model", self.args.openai_model])
        if hasattr(self.args, 'gemini_model') and self.args.gemini_model:
            cmd.extend(["--gemini_model", self.args.gemini_model])
        if hasattr(self.args, 'enable_thinking'):
            if self.args.enable_thinking:
                cmd.append("--enable_thinking")
            else:
                cmd.append("--disable_thinking")
        
        # Add W&B parameters
        cmd.extend([
            "--use_wandb",
            "--wandb_project", wandb_project,
            "--wandb_name", f"llm_baselines_{self.task_type}_task{task_id}_split{split_idx}",
        ])
        
        return cmd, eval_output_dir
    
    def evaluate_llm_baselines_on_task(self, task, split_idx: int) -> Optional[str]:
        """
        Evaluate LLM baselines on a specific OpenML task and split.
        
        Args:
            task: OpenML task object
            split_idx: Index of the split to use
            
        Returns:
            Path to the evaluation results or None if failed/skipped
        """
        task_id = task.task_id
        dataset_name = task.get_dataset().name
        
        self.logger.info(f"Evaluating LLM baselines on {self.task_type} task {task_id} ({dataset_name}), split {split_idx+1}/{self.args.num_splits}")
        
        # Build command
        result = self.build_evaluation_command(task, split_idx)
        if result is None:
            return None  # Skipped due to existing results
        
        cmd, eval_output_dir = result
        
        # Run evaluation command
        self.logger.info(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            self.logger.info(f"LLM baseline evaluation completed for {self.task_type} task {task_id}, split {split_idx+1}")
            return eval_output_dir
        except subprocess.CalledProcessError as e:
            self.logger.error(f"LLM baseline evaluation failed for {self.task_type} task {task_id}, split {split_idx+1}: {e}")
            return None
    
    def process_task(self, task) -> None:
        """
        Process a single task: evaluate LLM baselines on multiple splits.
        
        Args:
            task: OpenML task object
        """
        task_id = task.task_id
        dataset_name = task.get_dataset().name
        
        self.logger.info(f"Processing {self.task_type} task {task_id}: {dataset_name}")
        
        # Create task directory
        task_dir = os.path.join(self.args.output_dir, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)
        
        # Generate version information for tracking
        today = datetime.now()
        version_by_date = f"v{today.strftime('%Y%m%d')}"
        run_timestamp = today.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Save task metadata
        task_info = self._create_task_metadata(task, version_by_date, run_timestamp)
        with open(os.path.join(task_dir, f"task_info_{run_timestamp}.json"), "w") as f:
            json.dump(task_info, f, indent=2)
        
        # Process each split
        for split_idx in range(self.args.num_splits):
            # Evaluate LLM baselines
            eval_dir = self.evaluate_llm_baselines_on_task(task, split_idx)
            if eval_dir is None:
                self.logger.error(f"LLM baseline evaluation failed for {self.task_type} task {task_id}, split {split_idx+1}")
    
    def _create_task_metadata(self, task, version_by_date: str, run_timestamp: str) -> Dict[str, Any]:
        """Create metadata dictionary for a task."""
        task_info = {
            "task_id": task.task_id,
            "dataset_id": task.dataset_id,
            "dataset_name": task.get_dataset().name,
            "task_type": self.task_type,
            "version": version_by_date,
            "timestamp": run_timestamp,
            "models_evaluated": self.args.models,
            "evaluation_params": {
                "num_splits": self.args.num_splits,
                "num_few_shot_examples": self.args.num_few_shot_examples,
                "max_test_samples": self.args.max_test_samples,
                "device": self.args.device,
                "backend": self.args.backend,
                "tensor_parallel_size": self.args.tensor_parallel_size,
                "gpu_memory_utilization": self.args.gpu_memory_utilization,
                "clam_tsne_params": {
                    "vlm_model_id": self.args.vlm_model_id,
                    "embedding_size": self.args.embedding_size,
                    "tsne_perplexity": self.args.tsne_perplexity,
                    "tsne_max_iter": getattr(self.args, 'tsne_max_iter', getattr(self.args, 'tsne_n_iter', 1000)),
                    "max_tabpfn_samples": self.args.max_tabpfn_samples,
                    "use_3d": self.args.use_3d,
                    "viewing_angles": self.args.viewing_angles,
                    "use_knn_connections": self.args.use_knn_connections,
                    "nn_k": self.args.nn_k,
                    "max_vlm_image_size": self.args.max_vlm_image_size,
                    "image_dpi": self.args.image_dpi,
                    "force_rgb_mode": self.args.force_rgb_mode,
                    "save_sample_visualizations": self.args.save_sample_visualizations,
                    "visualization_save_cadence": self.args.visualization_save_cadence
                }
            }
        }
        
        # Add task-type specific metadata
        if self.task_type == "classification":
            task_info["num_classes"] = len(task.class_labels) if hasattr(task, "class_labels") else None
        elif self.task_type == "regression":
            task_info["target_attribute"] = task.target_name if hasattr(task, "target_name") else None
        
        # Add feature count if available
        if hasattr(task.get_dataset(), "features") and isinstance(task.get_dataset().features, dict):
            task_info["num_features"] = len(task.get_dataset().features)
        
        return task_info
    
    def aggregate_results(self) -> None:
        """
        Aggregate results from all tasks and splits into summary files.
        """
        self.logger.info(f"Aggregating results from all {self.task_type} tasks and splits")
        
        all_results = []
        summary_by_model = {}
        
        # Walk through all result directories
        for task_dir in os.listdir(self.args.output_dir):
            if not task_dir.startswith("task_"):
                continue
            
            task_path = os.path.join(self.args.output_dir, task_dir)
            if not os.path.isdir(task_path):
                continue
            
            task_id = task_dir.replace("task_", "")
            
            for split_dir in os.listdir(task_path):
                if not split_dir.startswith("split_"):
                    continue
                
                split_path = os.path.join(task_path, split_dir)
                if not os.path.isdir(split_path):
                    continue
                
                split_idx = split_dir.replace("split_", "")
                
                # Look for LLM baseline results
                llm_baselines_path = os.path.join(split_path, "llm_baselines")
                if not os.path.exists(llm_baselines_path):
                    continue
                
                # Check for aggregated results file
                aggregated_file = os.path.join(llm_baselines_path, "aggregated_results.json")
                if os.path.exists(aggregated_file):
                    try:
                        with open(aggregated_file, 'r') as f:
                            results = json.load(f)
                        
                        for result in results:
                            result['task_id'] = task_id
                            result['split_idx'] = split_idx
                            all_results.append(result)
                            
                            # Update summary by model
                            model_name = result.get('model_name', 'unknown')
                            if model_name not in summary_by_model:
                                summary_by_model[model_name] = self._init_model_summary()
                            
                            self._update_model_summary(summary_by_model[model_name], result)
                    
                    except Exception as e:
                        self.logger.error(f"Error reading results from {aggregated_file}: {e}")
        
        # Save aggregated results
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Save all results
        all_results_file = os.path.join(self.args.output_dir, f"all_{self.task_type}_results_{timestamp}.json")
        with open(all_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Calculate and save summary statistics
        summary_stats = self._calculate_summary_stats(summary_by_model)
        summary_file = os.path.join(self.args.output_dir, f"{self.task_type}_summary_stats_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Log summary
        self.logger.info(f"Aggregation complete. Found {len(all_results)} total {self.task_type} results.")
        self._log_summary_stats(summary_stats)
        
        self.logger.info(f"All results saved to: {all_results_file}")
        self.logger.info(f"Summary statistics saved to: {summary_file}")
    
    def _init_model_summary(self) -> Dict[str, Any]:
        """Initialize model summary structure based on task type."""
        if self.task_type == "classification":
            return {
                'accuracies': [],
                'balanced_accuracies': [],
                'total_tasks': 0,
                'successful_tasks': 0
            }
        elif self.task_type == "regression":
            return {
                'r2_scores': [],
                'mae_scores': [],
                'mse_scores': [],
                'total_tasks': 0,
                'successful_tasks': 0
            }
        else:
            return {
                'total_tasks': 0,
                'successful_tasks': 0
            }
    
    def _update_model_summary(self, summary: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Update model summary with a new result."""
        summary['total_tasks'] += 1
        
        if self.task_type == "classification":
            if 'accuracy' in result:
                summary['accuracies'].append(result['accuracy'])
                summary['successful_tasks'] += 1
            if 'balanced_accuracy' in result:
                summary['balanced_accuracies'].append(result['balanced_accuracy'])
        elif self.task_type == "regression":
            if 'r2_score' in result:
                summary['r2_scores'].append(result['r2_score'])
                summary['successful_tasks'] += 1
            if 'mae' in result:
                summary['mae_scores'].append(result['mae'])
            if 'mse' in result:
                summary['mse_scores'].append(result['mse'])
    
    def _calculate_summary_stats(self, summary_by_model: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate summary statistics for each model."""
        summary_stats = {}
        
        for model_name, stats in summary_by_model.items():
            if self.task_type == "classification" and stats['accuracies']:
                summary_stats[model_name] = {
                    'mean_accuracy': np.mean(stats['accuracies']),
                    'std_accuracy': np.std(stats['accuracies']),
                    'mean_balanced_accuracy': np.mean(stats['balanced_accuracies']) if stats['balanced_accuracies'] else None,
                    'std_balanced_accuracy': np.std(stats['balanced_accuracies']) if stats['balanced_accuracies'] else None,
                    'total_evaluations': stats['total_tasks'],
                    'successful_evaluations': stats['successful_tasks'],
                    'success_rate': stats['successful_tasks'] / stats['total_tasks'] if stats['total_tasks'] > 0 else 0
                }
            elif self.task_type == "regression" and stats['r2_scores']:
                summary_stats[model_name] = {
                    'mean_r2_score': np.mean(stats['r2_scores']),
                    'std_r2_score': np.std(stats['r2_scores']),
                    'mean_mae': np.mean(stats['mae_scores']) if stats['mae_scores'] else None,
                    'std_mae': np.std(stats['mae_scores']) if stats['mae_scores'] else None,
                    'mean_mse': np.mean(stats['mse_scores']) if stats['mse_scores'] else None,
                    'std_mse': np.std(stats['mse_scores']) if stats['mse_scores'] else None,
                    'total_evaluations': stats['total_tasks'],
                    'successful_evaluations': stats['successful_tasks'],
                    'success_rate': stats['successful_tasks'] / stats['total_tasks'] if stats['total_tasks'] > 0 else 0
                }
        
        return summary_stats
    
    def _log_summary_stats(self, summary_stats: Dict[str, Dict[str, Any]]) -> None:
        """Log summary statistics."""
        for model_name, stats in summary_stats.items():
            if self.task_type == "classification":
                self.logger.info(f"{model_name}: Mean accuracy = {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f} "
                               f"({stats['successful_evaluations']}/{stats['total_evaluations']} successful)")
            elif self.task_type == "regression":
                self.logger.info(f"{model_name}: Mean R² = {stats['mean_r2_score']:.4f} ± {stats['std_r2_score']:.4f}")
                if stats['mean_mae']:
                    self.logger.info(f"  Mean MAE = {stats['mean_mae']:.4f} ± {stats['std_mae']:.4f}")
                self.logger.info(f"  ({stats['successful_evaluations']}/{stats['total_evaluations']} successful)")


def handle_metadata_validation(args, tasks: List[Any], models_to_check: List[str], task_type: str = "classification") -> List[Any]:
    """
    Handle metadata validation and filtering for OpenML tasks.
    
    Args:
        args: Command line arguments
        tasks: List of OpenML task objects
        models_to_check: List of model names to validate
        task_type: Type of tasks ("classification" or "regression")
        
    Returns:
        Filtered list of tasks
    """
    from clam.utils.metadata_validation import generate_metadata_coverage_report, print_metadata_coverage_report
    
    logger = logging.getLogger(__name__)
    
    # Handle metadata validation
    if args.validate_metadata_only:
        logger.info(f"Running metadata validation only for {task_type} tasks...")
        task_ids = [task.task_id for task in tasks]
        report = generate_metadata_coverage_report(task_ids, models_to_check)
        print_metadata_coverage_report(report)
        
        # Save detailed report
        report_file = os.path.join(args.output_dir, f"{task_type}_metadata_coverage_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Detailed metadata report saved to: {report_file}")
        return []  # Return empty list to signal exit
    
    # Filter tasks based on metadata availability if requested
    if args.skip_missing_metadata:
        logger.info(f"Filtering {task_type} tasks based on metadata availability...")
        task_ids = [task.task_id for task in tasks]
        report = generate_metadata_coverage_report(task_ids, models_to_check)
        
        # Keep only tasks where at least one model has valid metadata
        valid_task_ids = []
        for task_id, results in report['detailed_results'].items():
            if any(result['valid'] for result in results.values()):
                valid_task_ids.append(task_id)
        
        # Filter tasks list
        original_count = len(tasks)
        tasks = [task for task in tasks if task.task_id in valid_task_ids]
        logger.info(f"Filtered {original_count} {task_type} tasks to {len(tasks)} tasks with valid metadata")
        
        if len(tasks) == 0:
            logger.error(f"No {task_type} tasks have valid metadata for any of the requested models. Exiting.")
            return []
    
    return tasks


def generate_metadata_report(tasks: List[Any], args, task_type: str = "classification") -> List[Dict[str, Any]]:
    """
    Generate a metadata coverage report for the tasks.
    
    Args:
        tasks: List of OpenML task objects
        args: Command line arguments
        task_type: Type of tasks ("classification" or "regression")
        
    Returns:
        List of task summary dictionaries
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating metadata coverage report for {task_type} tasks...")
    
    # Create a summary of tasks and their metadata
    task_summary = []
    for task in tasks:
        try:
            dataset = task.get_dataset()
            task_info = {
                "task_id": task.task_id,
                "dataset_name": dataset.name,
                "task_type": task.task_type,
                "num_features": len(dataset.features) if hasattr(dataset, "features") and isinstance(dataset.features, dict) else 0,
                "num_instances": dataset.qualities.get("NumberOfInstances", 0) if hasattr(dataset, "qualities") else 0
            }
            
            # Add task-type specific info
            if task_type == "classification":
                task_info["num_classes"] = len(task.class_labels) if hasattr(task, "class_labels") else None
            elif task_type == "regression":
                task_info["target_attribute"] = task.target_name if hasattr(task, "target_name") else "unknown"
            
            task_summary.append(task_info)
        except Exception as e:
            logger.warning(f"Could not extract metadata for task {task.task_id}: {e}")
    
    # Save metadata report
    report_path = os.path.join(args.output_dir, f"{task_type}_tasks_metadata_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "total_tasks": len(tasks),
            "task_type": task_type,
            "report_timestamp": datetime.now().isoformat(),
            "tasks": task_summary
        }, f, indent=2)
    
    logger.info(f"Metadata report saved to {report_path}")
    return task_summary