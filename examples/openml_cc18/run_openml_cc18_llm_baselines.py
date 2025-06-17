#!/usr/bin/env python
"""
Script to evaluate LLM baselines (TabLLM, Tabula-8B, JOLT, and LlaTa-T-SNe) on the OpenML CC18 collection.

This script:
1. Retrieves the OpenML CC18 collection (study_id=99)
2. For each task in the collection:
   a. Evaluates TabLLM, Tabula-8B, JOLT, and LlaTa-T-SNe baselines on multiple splits
3. Logs the results to Weights & Biases with version control by date

Requirements:
- OpenML installed (pip install openml)
- LLATA installed and configured
- W&B account for logging results
- RTFM package for Tabula-8B (pip install git+https://github.com/penfever/rtfm.git)
- Transformers and torch for LLM baselines
- Vision dependencies for LlaTa-T-SNe: PIL, scikit-learn, matplotlib

Usage:
    # Basic usage with all models
    python run_openml_cc18_llm_baselines.py --llata_repo_path /path/to/llata --output_dir ./results
    
    # Run only LlaTa-T-SNe with 3D t-SNE and KNN connections
    python run_openml_cc18_llm_baselines.py --llata_repo_path /path/to/llata --output_dir ./results \
        --models llata_tsne --use_3d_tsne --use_knn_connections --knn_k 7
    
    # Run with custom image settings
    python run_openml_cc18_llm_baselines.py --llata_repo_path /path/to/llata --output_dir ./results \
        --models llata_tsne --max_vlm_image_size 1024 --image_dpi 72 --no-force_rgb_mode

The script assumes the LLATA repo structure.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openml_cc18_llm_baselines.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM baselines on OpenML CC18 collection")
    
    parser.add_argument(
        "--llata_repo_path",
        type=str,
        required=True,
        help="Path to the LLATA repository"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./openml_cc18_llm_baselines_results",
        help="Directory to save all results"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="tabllm,tabula_8b,jolt,llata_tsne",
        help="Comma-separated list of LLM models to evaluate: 'tabllm', 'tabula_8b', 'jolt', 'llata_tsne'"
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=3,
        help="Number of different train/test splits to use for each task"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="llata-openml-cc18-llm-baselines",
        help="W&B project name"
    )
    parser.add_argument(
        "--task_ids",
        type=str,
        default=None,
        help="Comma-separated list of task IDs to run (default: all tasks in CC18)"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start from this task index in the CC18 collection"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End at this task index in the CC18 collection (exclusive)"
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Maximum number of test samples to use for evaluation (to speed up)"
    )
    parser.add_argument(
        "--num_few_shot_examples",
        type=int,
        default=16,
        help="Number of few-shot examples to include in prompts"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('cuda', 'cpu', or 'auto')"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "vllm", "transformers"],
        help="Backend to use for model loading (auto chooses VLLM if available)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism in VLLM"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for VLLM (0.0-1.0)"
    )
    
    # LlaTa-T-SNe specific parameters
    parser.add_argument(
        "--vlm_model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-32B-Instruct",
        help="HuggingFace model name for Vision Language Model (LlaTa-T-SNe baseline)"
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=1000,
        help="Size of TabPFN embeddings for LlaTa-T-SNe baseline"
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity parameter for LlaTa-T-SNe baseline"
    )
    parser.add_argument(
        "--tsne_n_iter",
        type=int,
        default=1000,
        help="Number of t-SNE iterations for LlaTa-T-SNe baseline"
    )
    parser.add_argument(
        "--max_tabpfn_samples",
        type=int,
        default=3000,
        help="Maximum samples for TabPFN fitting in LlaTa-T-SNe baseline"
    )
    parser.add_argument(
        "--use_3d_tsne",
        action="store_true",
        help="Use 3D t-SNE with multiple viewing angles instead of 2D (LlaTa-T-SNe baseline)"
    )
    parser.add_argument(
        "--viewing_angles",
        type=str,
        default=None,
        help="Custom viewing angles for 3D t-SNE as 'elev1,azim1;elev2,azim2;...' (LlaTa-T-SNe baseline)"
    )
    parser.add_argument(
        "--use_knn_connections",
        action="store_true",
        help="Show KNN connections from query point to nearest neighbors in embedding space (LlaTa-T-SNe baseline)"
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=5,
        help="Number of nearest neighbors to show when using KNN connections (LlaTa-T-SNe baseline)"
    )
    parser.add_argument(
        "--max_vlm_image_size",
        type=int,
        default=2048,
        help="Maximum image size (width/height) for VLM compatibility (LlaTa-T-SNe baseline)"
    )
    parser.add_argument(
        "--image_dpi",
        type=int,
        default=100,
        help="DPI for saving t-SNE visualizations (LlaTa-T-SNe baseline)"
    )
    parser.add_argument(
        "--force_rgb_mode",
        action="store_true",
        default=True,
        help="Convert images to RGB mode to improve VLM processing speed (LlaTa-T-SNe baseline)"
    )
    parser.add_argument(
        "--no-force_rgb_mode",
        action="store_false",
        dest="force_rgb_mode",
        help="Disable RGB conversion (keep RGBA mode) for LlaTa-T-SNe baseline"
    )
    parser.add_argument(
        "--save_sample_visualizations",
        action="store_true",
        default=True,
        help="Save sample t-SNE visualizations for debugging and documentation (LlaTa-T-SNe baseline)"
    )
    parser.add_argument(
        "--no-save_sample_visualizations",
        action="store_false",
        dest="save_sample_visualizations",
        help="Disable saving of sample t-SNE visualizations (LlaTa-T-SNe baseline)"
    )
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_openml_cc18_tasks():
    """
    Get the list of tasks in the OpenML CC18 collection (study_id=99).
    
    Returns:
        List of OpenML task objects
    """
    logger.info("Fetching OpenML CC18 collection (study_id=99)")
    
    try:
        # Try the newer API method first
        logger.info("Attempting to fetch CC18 using newer API method (get_suite)")
        suite = openml.study.get_suite(99)  # 99 is the ID for CC18
        task_ids = suite.tasks
    except Exception as e1:
        logger.warning(f"Error using get_suite: {e1}")
        try:
            # Try fallback method
            logger.info("Attempting fallback method with get_study and entity_type='task'")
            study = openml.study.functions._get_study(99, entity_type='task')
            task_ids = study.tasks
        except Exception as e2:
            logger.warning(f"Error using get_study fallback: {e2}")
            # Hardcoded list of CC18 tasks as a last resort
            logger.info("Using hardcoded list of CC18 tasks")
            task_ids = [
                3573, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3917, 3918,
                3950, 3954, 7592, 7593, 9914, 9946, 9957, 9960, 9961, 9962, 9964, 9965, 9966, 9967, 9968,
                9970, 9971, 9976, 9977, 9978, 9979, 9980, 9981, 9983, 9985, 9986, 9987, 10060, 10061,
                10064, 10065, 10066, 10067, 10068, 10069, 10070, 10071, 10072, 10073, 10074, 10075, 10076,
                10077, 10078, 10079, 10080, 10081, 10082, 10083, 10084, 10085, 10086, 10087, 10088, 10089,
                10090, 10092, 10093, 10096, 10097, 10098, 10099, 10100, 10101, 14954, 14965, 14969, 14970,
                125920, 125921, 125922, 125923, 125928, 125929, 125920, 125921, 125922, 125923, 125928,
                125929, 125930, 125931, 125932, 125933, 125934, 14954, 14965, 14969, 14970, 34536, 34537,
                34539, 146574
            ]
            # Remove duplicates
            task_ids = list(set(task_ids))

    logger.info(f"Retrieved {len(task_ids)} tasks from CC18 collection")
    
    tasks = []
    for task_id in task_ids:
        try:
            task = openml.tasks.get_task(task_id)
            tasks.append(task)
            logger.info(f"Retrieved task {task_id}: {task.get_dataset().name}")
        except Exception as e:
            logger.error(f"Error retrieving task {task_id}: {e}")
    
    logger.info(f"Successfully retrieved {len(tasks)} tasks")
    return tasks

def evaluate_llm_baselines_on_task(task, split_idx, args):
    """
    Evaluate LLM baselines on a specific OpenML task and split.
    
    Args:
        task: OpenML task object
        split_idx: Index of the split to use
        args: Command line arguments
    
    Returns:
        Path to the evaluation results
    """
    task_id = task.task_id
    dataset_id = task.dataset_id
    dataset_name = task.get_dataset().name
    
    logger.info(f"Evaluating LLM baselines on task {task_id} ({dataset_name}), split {split_idx+1}/{args.num_splits}")
    
    # Create output directory
    eval_output_dir = os.path.join(
        args.output_dir, 
        f"task_{task_id}", 
        f"split_{split_idx}", 
        "llm_baselines"
    )
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Generate version tag based on date for W&B project
    today = datetime.now()
    version_by_date = f"v{today.strftime('%Y%m%d')}"
    wandb_project = f"{args.wandb_project}-{version_by_date}"
    
    # Build evaluation command using the LLM baselines script
    eval_script = os.path.join(args.llata_repo_path, "examples", "evaluate_llm_baselines.py")
    
    cmd = [
        "python", eval_script,
        "--dataset_name", str(dataset_id),  # Pass dataset_id as dataset_name
        "--output_dir", eval_output_dir,
        "--models", args.models,
        "--num_few_shot_examples", str(args.num_few_shot_examples),
        "--seed", str(args.seed + split_idx),  # Vary seed for different splits
        "--device", args.device,
    ]
    
    # Add optional parameters
    if args.max_test_samples:
        cmd.extend(["--max_test_samples", str(args.max_test_samples)])
    
    # Add backend parameters
    cmd.extend([
        "--backend", args.backend,
        "--tensor_parallel_size", str(args.tensor_parallel_size),
        "--gpu_memory_utilization", str(args.gpu_memory_utilization),
    ])
    
    # Add LlaTa-T-SNe specific parameters
    cmd.extend([
        "--vlm_model_id", args.vlm_model_id,
        "--embedding_size", str(args.embedding_size),
        "--tsne_perplexity", str(args.tsne_perplexity),
        "--tsne_n_iter", str(args.tsne_n_iter),
        "--max_tabpfn_samples", str(args.max_tabpfn_samples),
        "--knn_k", str(args.knn_k),
        "--max_vlm_image_size", str(args.max_vlm_image_size),
        "--image_dpi", str(args.image_dpi),
    ])
    
    # Add LlaTa-T-SNe boolean flags
    if args.use_3d_tsne:
        cmd.append("--use_3d_tsne")
    if args.use_knn_connections:
        cmd.append("--use_knn_connections")
    if args.force_rgb_mode:
        cmd.append("--force_rgb_mode")
    else:
        cmd.append("--no-force_rgb_mode")
    if args.save_sample_visualizations:
        cmd.append("--save_sample_visualizations")
    else:
        cmd.append("--no-save_sample_visualizations")
    
    # Add custom viewing angles if specified
    if args.viewing_angles:
        cmd.extend(["--viewing_angles", args.viewing_angles])
    
    # Add W&B parameters if we want to track (commented out by default to avoid clutter)
    cmd.extend([
        "--use_wandb",
        "--wandb_project", wandb_project,
        "--wandb_name", f"llm_baselines_task{task_id}_split{split_idx}",
    ])
    
    # Run evaluation command
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"LLM baseline evaluation completed for task {task_id}, split {split_idx+1}")
        return eval_output_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"LLM baseline evaluation failed for task {task_id}, split {split_idx+1}: {e}")
        return None

def process_task(task, args):
    """
    Process a single task: evaluate LLM baselines on multiple splits.
    
    Args:
        task: OpenML task object
        args: Command line arguments
    """
    task_id = task.task_id
    dataset_name = task.get_dataset().name
    
    logger.info(f"Processing task {task_id}: {dataset_name}")
    
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
            "num_classes": len(task.class_labels) if hasattr(task, "class_labels") else None,
            "num_features": len(task.get_dataset().features) if hasattr(task.get_dataset(), "features") and isinstance(task.get_dataset().features, dict) else None,
            "version": version_by_date,
            "timestamp": run_timestamp,
            "evaluation_params": {
                "models": args.models,
                "num_few_shot_examples": args.num_few_shot_examples,
                "max_test_samples": args.max_test_samples,
                "device": args.device,
                "backend": args.backend,
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "llata_tsne_params": {
                    "vlm_model_id": args.vlm_model_id,
                    "embedding_size": args.embedding_size,
                    "tsne_perplexity": args.tsne_perplexity,
                    "tsne_n_iter": args.tsne_n_iter,
                    "max_tabpfn_samples": args.max_tabpfn_samples,
                    "use_3d_tsne": args.use_3d_tsne,
                    "viewing_angles": args.viewing_angles,
                    "use_knn_connections": args.use_knn_connections,
                    "knn_k": args.knn_k,
                    "max_vlm_image_size": args.max_vlm_image_size,
                    "image_dpi": args.image_dpi,
                    "force_rgb_mode": args.force_rgb_mode,
                    "save_sample_visualizations": args.save_sample_visualizations
                }
            }
        }
        json.dump(task_info, f, indent=2)
    
    # Process each split
    for split_idx in range(args.num_splits):
        # Evaluate LLM baselines
        eval_dir = evaluate_llm_baselines_on_task(task, split_idx, args)
        if eval_dir is None:
            logger.error(f"LLM baseline evaluation failed for task {task_id}, split {split_idx+1}")

def aggregate_results(args):
    """
    Aggregate results from all tasks and splits into summary files.
    
    Args:
        args: Command line arguments
    """
    logger.info("Aggregating results from all tasks and splits")
    
    all_results = []
    summary_by_model = {}
    
    # Walk through all result directories
    for task_dir in os.listdir(args.output_dir):
        if not task_dir.startswith("task_"):
            continue
        
        task_path = os.path.join(args.output_dir, task_dir)
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
                            summary_by_model[model_name] = {
                                'accuracies': [],
                                'balanced_accuracies': [],
                                'total_tasks': 0,
                                'successful_tasks': 0
                            }
                        
                        summary_by_model[model_name]['total_tasks'] += 1
                        if 'accuracy' in result:
                            summary_by_model[model_name]['accuracies'].append(result['accuracy'])
                            summary_by_model[model_name]['successful_tasks'] += 1
                        if 'balanced_accuracy' in result:
                            summary_by_model[model_name]['balanced_accuracies'].append(result['balanced_accuracy'])
                
                except Exception as e:
                    logger.error(f"Error reading results from {aggregated_file}: {e}")
    
    # Save aggregated results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save all results
    all_results_file = os.path.join(args.output_dir, f"all_results_{timestamp}.json")
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Calculate and save summary statistics
    summary_stats = {}
    for model_name, stats in summary_by_model.items():
        if stats['accuracies']:
            summary_stats[model_name] = {
                'mean_accuracy': np.mean(stats['accuracies']),
                'std_accuracy': np.std(stats['accuracies']),
                'mean_balanced_accuracy': np.mean(stats['balanced_accuracies']) if stats['balanced_accuracies'] else None,
                'std_balanced_accuracy': np.std(stats['balanced_accuracies']) if stats['balanced_accuracies'] else None,
                'total_evaluations': stats['total_tasks'],
                'successful_evaluations': stats['successful_tasks'],
                'success_rate': stats['successful_tasks'] / stats['total_tasks'] if stats['total_tasks'] > 0 else 0
            }
    
    summary_file = os.path.join(args.output_dir, f"summary_stats_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Log summary
    logger.info(f"Aggregation complete. Found {len(all_results)} total results.")
    for model_name, stats in summary_stats.items():
        logger.info(f"{model_name}: Mean accuracy = {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f} "
                   f"({stats['successful_evaluations']}/{stats['total_evaluations']} successful)")
    
    logger.info(f"All results saved to: {all_results_file}")
    logger.info(f"Summary statistics saved to: {summary_file}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Get OpenML CC18 tasks
    tasks = get_openml_cc18_tasks()
    
    # Filter tasks if task_ids is provided
    if args.task_ids:
        task_ids = [int(id.strip()) for id in args.task_ids.split(",")]
        tasks = [task for task in tasks if task.task_id in task_ids]
        logger.info(f"Filtered to {len(tasks)} specified tasks")
    
    # Apply start and end indices
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(tasks)
    tasks = tasks[start_idx:end_idx]
    logger.info(f"Processing tasks from index {start_idx} to {end_idx} (total: {len(tasks)})")
    
    # Process each task
    for i, task in enumerate(tasks):
        try:
            logger.info(f"Processing task {i+1}/{len(tasks)}")
            process_task(task, args)
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
    
    # Aggregate results at the end
    try:
        aggregate_results(args)
    except Exception as e:
        logger.error(f"Error aggregating results: {e}")
    
    logger.info("All tasks completed")

if __name__ == "__main__":
    main()