#!/usr/bin/env python
"""
Script to evaluate LLM baselines (TabLLM, Tabula-8B, JOLT, and CLAM-T-SNe) on the New OpenML Suite 2025 regression collection.

This script:
1. Retrieves the New OpenML Suite 2025 regression collection (study_id=455)
2. For each task in the collection:
   a. Evaluates TabLLM, Tabula-8B, JOLT, and CLAM-T-SNe baselines on multiple splits
3. Logs the results to Weights & Biases with version control by date

Requirements:
- OpenML installed (pip install openml)
- CLAM installed and configured
- W&B account for logging results
- RTFM package for Tabula-8B (pip install git+https://github.com/penfever/rtfm.git)
- Transformers and torch for LLM baselines
- Vision dependencies for CLAM-T-SNe: PIL, scikit-learn, matplotlib

Usage:
    # Basic usage with all models
    python run_openml_regression_2025_llm_baselines.py --clam_repo_path /path/to/clam --output_dir ./results
    
    # Run only CLAM-T-SNe with 3D t-SNE and KNN connections
    python run_openml_regression_2025_llm_baselines.py --clam_repo_path /path/to/clam --output_dir ./results \
        --models clam_tsne --use_3d --use_knn_connections --nn_k 7
    
    # Run with custom image settings for regression visualization
    python run_openml_regression_2025_llm_baselines.py --clam_repo_path /path/to/clam --output_dir ./results \
        --models clam_tsne --max_vlm_image_size 1024 --image_dpi 72 --no-force_rgb_mode

The script assumes the CLAM repo structure.
"""

import os
import argparse
import subprocess
import json
import logging
import openml
from pathlib import Path
import random
import numpy as np
import torch
import time
from datetime import datetime
from tqdm import tqdm

# Import centralized argument parser
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from clam.utils.evaluation_args import create_tabular_llm_evaluation_parser
from clam.utils.metadata_validation import generate_metadata_coverage_report, print_metadata_coverage_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openml_regression_2025_llm_baselines.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments using centralized tabular LLM evaluation parser."""
    parser = create_tabular_llm_evaluation_parser("Evaluate LLM baselines on New OpenML Suite 2025 regression collection")
    
    # Remove the dataset source requirement since we automatically fetch regression tasks
    # Find the mutually exclusive group and make it not required
    for action_group in parser._mutually_exclusive_groups:
        if any(action.dest in ['dataset_name', 'dataset_ids', 'data_dir', 'num_datasets'] 
               for action in action_group._group_actions):
            action_group.required = False
            break
    
    # Add OpenML regression 2025 orchestration-specific arguments
    parser.add_argument(
        "--clam_repo_path",
        type=str,
        required=True,
        help="Path to the CLAM repository"
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=3,
        help="Number of different train/test splits to use for each task"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start from this task index in the regression collection"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End at this task index in the regression collection (exclusive)"
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Force rerun even if output files already exist"
    )
    parser.add_argument(
        "--save_detailed_outputs",
        action="store_true",
        help="Save detailed VLM outputs and visualizations"
    )
    
    # Override some defaults for OpenML regression 2025 context
    parser.set_defaults(
        output_dir="./openml_regression_2025_llm_results",
        wandb_project="clam-regression-llm-baselines-2025",
        models=["clam_tsne", "tabllm", "jolt"],  # Remove tabula_8b for now as it may need adaptation
        model_id="Qwen/Qwen2.5-3B-Instruct",
        nn_k=7,
        use_3d=True
    )
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_openml_regression_2025_tasks():
    """
    Get the list of tasks in the New OpenML Suite 2025 regression collection (study_id=455).
    
    Returns:
        List of OpenML task objects
    """
    logger.info("Fetching New OpenML Suite 2025 regression collection (study_id=455)")
    
    try:
        # Try the newer API method first
        logger.info("Attempting to fetch regression suite using newer API method (get_suite)")
        suite = openml.study.get_suite(455)  # 455 is the ID for New_OpenML_Suite_2025_regression
        task_ids = suite.tasks
    except Exception as e1:
        logger.warning(f"Error using get_suite: {e1}")
        try:
            # Try fallback method
            logger.info("Attempting fallback method with get_study and entity_type='task'")
            study = openml.study.functions._get_study(455, entity_type='task')
            task_ids = study.tasks
        except Exception as e2:
            logger.warning(f"Error using get_study fallback: {e2}")
            # If both methods fail, we'll need to fetch individual tasks or use a hardcoded list
            logger.error("Could not fetch regression suite. Please check study_id=455 exists and contains regression tasks.")
            return []

    logger.info(f"Retrieved {len(task_ids)} task IDs from regression collection")
    
    tasks = []
    for task_id in task_ids:
        try:
            task = openml.tasks.get_task(task_id)
            # Verify this is a regression task
            if task.task_type.lower() != 'supervised regression':
                logger.warning(f"Task {task_id} is not a regression task (type: {task.task_type}), skipping")
                continue
            tasks.append(task)
            logger.info(f"Retrieved regression task {task_id}: {task.get_dataset().name}")
        except Exception as e:
            logger.error(f"Error retrieving task {task_id}: {e}")
    
    logger.info(f"Successfully retrieved {len(tasks)} regression tasks")
    return tasks

def evaluate_baseline_on_task(task, split_idx, model_name, args):
    """
    Evaluate a baseline model on a specific OpenML regression task and split.
    
    Args:
        task: OpenML task object
        split_idx: Index of the split to use
        model_name: Name of the baseline model to evaluate
        args: Command line arguments
    
    Returns:
        Path to the evaluation results
    """
    task_id = task.task_id
    dataset_id = task.dataset_id
    dataset_name = task.get_dataset().name
    
    logger.info(f"Evaluating {model_name} on regression task {task_id} ({dataset_name}), split {split_idx+1}/{args.num_splits}")
    
    # Create output directory
    eval_output_dir = os.path.join(
        args.output_dir, 
        f"task_{task_id}", 
        f"split_{split_idx}", 
        f"{model_name}_results"
    )
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Check if results already exist and skip if not forcing rerun
    results_file = os.path.join(eval_output_dir, "aggregated_results.json")
    if os.path.exists(results_file) and not args.force_rerun:
        logger.info(f"Results already exist for {model_name} on task {task_id}, split {split_idx+1}. Skipping.")
        return eval_output_dir
    
    # Generate version tag based on date for W&B project
    today = datetime.now()
    version_by_date = f"v{today.strftime('%Y%m%d')}"
    wandb_project = f"{args.wandb_project}-{version_by_date}"
    
    # Build evaluation command based on model type
    eval_script = os.path.join(args.clam_repo_path, "examples", "tabular", "evaluate_llm_baselines_tabular.py")
    
    base_cmd = [
        "python", eval_script,
        "--task_ids", str(task_id),  # Pass task_id properly for regression
        "--output_dir", eval_output_dir,
        "--models", model_name,
        "--use_wandb",
        "--wandb_entity", "nyu-dice-lab",
        "--wandb_project", wandb_project,
        "--wandb_name", f"{model_name}_regression_task{task_id}_split{split_idx}",
        "--seed", str(args.seed + split_idx),  # Use consistent seed
        "--feature_selection_threshold", str(args.feature_selection_threshold)
    ]
    
    # Add model-specific arguments
    if model_name == "clam_tsne":
        base_cmd.extend([
            "--model_id", args.model_id,
            "--nn_k", str(args.nn_k),
            "--max_vlm_image_size", str(args.max_vlm_image_size),
            "--image_dpi", str(args.image_dpi)
        ])
        
        if args.use_3d:
            base_cmd.append("--use_3d")
        if args.use_knn_connections:
            base_cmd.append("--use_knn_connections")
        if not args.force_rgb_mode:
            base_cmd.append("--no-force_rgb_mode")
        if args.save_detailed_outputs:
            base_cmd.extend([
                "--save_outputs",
                "--visualization_save_cadence", "10"
            ])
    
    elif model_name == "tabllm":
        base_cmd.extend([
            "--model_id", args.model_id
        ])
    
    elif model_name == "jolt":
        base_cmd.extend([
            "--model_id", args.model_id
        ])
    
    elif model_name == "tabula_8b":
        base_cmd.extend([
            "--model_id", "approximatelabs/tabula-8b"
        ])
    
    # Run evaluation command
    logger.info(f"Running command: {' '.join(base_cmd)}")
    try:
        subprocess.run(base_cmd, check=True)
        logger.info(f"{model_name} evaluation completed for regression task {task_id}, split {split_idx+1}")
        return eval_output_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"{model_name} evaluation failed for regression task {task_id}, split {split_idx+1}: {e}")
        return None

def process_task(task, args):
    """
    Process a single regression task: evaluate all baselines on multiple splits.
    
    Args:
        task: OpenML task object
        args: Command line arguments
    """
    task_id = task.task_id
    dataset_name = task.get_dataset().name
    
    logger.info(f"Processing regression task {task_id}: {dataset_name}")
    
    # Create task directory
    task_dir = os.path.join(args.output_dir, f"task_{task_id}")
    os.makedirs(task_dir, exist_ok=True)
    
    # Generate version information for tracking
    today = datetime.now()
    version_by_date = f"v{today.strftime('%Y%m%d')}"
    run_timestamp = today.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save task metadata
    with open(os.path.join(task_dir, f"task_info_{run_timestamp}.json"), "w") as f:
        task_info = {
            "task_id": task_id,
            "dataset_id": task.dataset_id,
            "dataset_name": dataset_name,
            "task_type": "regression",
            "target_attribute": task.target_name if hasattr(task, "target_name") else None,
            "num_features": len(task.get_dataset().features) if hasattr(task.get_dataset(), "features") and isinstance(task.get_dataset().features, dict) else None,
            "version": version_by_date,
            "timestamp": run_timestamp,
            "models_evaluated": args.models,
            "evaluation_params": {
                "num_splits": args.num_splits,
                "nn_k": args.nn_k,
                "use_3d": args.use_3d,
                "use_knn_connections": args.use_knn_connections,
                "max_vlm_image_size": args.max_vlm_image_size,
                "seed": args.seed
            }
        }
        json.dump(task_info, f, indent=2)
    
    # Process each split and model combination
    for split_idx in range(args.num_splits):
        for model_name in args.models:
            try:
                eval_dir = evaluate_baseline_on_task(task, split_idx, model_name, args)
                if eval_dir is None:
                    logger.error(f"Evaluation failed for {model_name} on regression task {task_id}, split {split_idx+1}")
            except Exception as e:
                logger.error(f"Error evaluating {model_name} on regression task {task_id}, split {split_idx+1}: {e}")

def generate_metadata_report(tasks, args):
    """Generate a metadata coverage report for the regression tasks."""
    logger.info("Generating metadata coverage report for regression tasks...")
    
    # Create a summary of tasks and their metadata
    task_summary = []
    for task in tasks:
        try:
            dataset = task.get_dataset()
            task_summary.append({
                "task_id": task.task_id,
                "dataset_name": dataset.name,
                "task_type": task.task_type,
                "target_attribute": task.target_name if hasattr(task, "target_name") else "unknown",
                "num_features": len(dataset.features) if hasattr(dataset, "features") and isinstance(dataset.features, dict) else 0,
                "num_instances": dataset.qualities.get("NumberOfInstances", 0) if hasattr(dataset, "qualities") else 0
            })
        except Exception as e:
            logger.warning(f"Could not extract metadata for task {task.task_id}: {e}")
    
    # Save metadata report
    report_path = os.path.join(args.output_dir, "regression_tasks_metadata_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "total_tasks": len(tasks),
            "report_timestamp": datetime.now().isoformat(),
            "tasks": task_summary
        }, f, indent=2)
    
    logger.info(f"Metadata report saved to {report_path}")
    return task_summary

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Get OpenML regression 2025 tasks
    tasks = get_openml_regression_2025_tasks()
    
    if not tasks:
        logger.error("No regression tasks found. Exiting.")
        return
    
    # Generate metadata report
    generate_metadata_report(tasks, args)
    
    # Filter tasks if task_ids is provided
    if args.task_ids:
        task_ids = [int(id.strip()) for id in args.task_ids.split(",")]
        tasks = [task for task in tasks if task.task_id in task_ids]
        logger.info(f"Filtered to {len(tasks)} specified regression tasks")
    
    # Apply start and end indices
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(tasks)
    tasks = tasks[start_idx:end_idx]
    logger.info(f"Processing regression tasks from index {start_idx} to {end_idx} (total: {len(tasks)})")
    
    # Process each task
    for i, task in enumerate(tasks):
        try:
            logger.info(f"Processing regression task {i+1}/{len(tasks)}")
            process_task(task, args)
        except Exception as e:
            logger.error(f"Error processing regression task {task.task_id}: {e}")
    
    logger.info("All regression tasks completed")

if __name__ == "__main__":
    main()