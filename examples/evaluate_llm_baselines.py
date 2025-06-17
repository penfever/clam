#!/usr/bin/env python
"""
Script for evaluating LLM baselines (TabLLM, Tabula-8B, JOLT, and LlaTa-T-SNe) on tabular datasets.
This script handles:
1. Loading and preprocessing datasets from multiple sources
2. Creating textual serializations for LLMs
3. Evaluating the four LLM baselines on these datasets
4. Optional Weights & Biases logging and visualization

LLM Baselines evaluated:
- TabLLM: Few-shot classification using textual serializations
- Tabula-8B: Large language model fine-tuned for tabular data
- JOLT: Joint probabilistic predictions on tabular data using LLMs
- LlaTa-T-SNe: Vision Language Model classification using t-SNE visualizations of TabPFN embeddings

Usage examples:
    # Basic usage with a single dataset
    python evaluate_llm_baselines.py --dataset_name har --output_dir ./llm_baseline_results
    
    # Evaluating on multiple specific datasets
    python evaluate_llm_baselines.py --dataset_ids 1590,40975,37,54 --output_dir ./llm_baseline_results
    
    # Evaluating on 5 randomly sampled datasets from OpenML
    python evaluate_llm_baselines.py --num_datasets 5 --output_dir ./llm_baseline_results
    
    # Evaluating only specific LLM models
    python evaluate_llm_baselines.py --dataset_name har --models tabula_8b,jolt,llata_tsne --output_dir ./results
    
    # Using Weights & Biases for experiment tracking
    python evaluate_llm_baselines.py --dataset_ids 1590,40975 --use_wandb --wandb_project llm_baselines
    
    # TabLLM automatically uses semantic templates and meaningful class names when available
    python evaluate_llm_baselines.py --dataset_name adult --models tabllm
    
    # Using 3D t-SNE with multiple viewing angles for LlaTa-T-SNe
    python evaluate_llm_baselines.py --dataset_name diabetes --models llata_tsne --use_3d_tsne --output_dir ./results
    
    # Custom viewing angles for 3D t-SNE
    python evaluate_llm_baselines.py --dataset_name har --models llata_tsne --use_3d_tsne --viewing_angles "20,45;0,0;90,0" --output_dir ./results
    
    # Using KNN connections to show nearest neighbors in embedding space
    python evaluate_llm_baselines.py --dataset_name adult --models llata_tsne --use_knn_connections --knn_k 7 --output_dir ./results
    
    # Combining 3D t-SNE with KNN connections for maximum information
    python evaluate_llm_baselines.py --dataset_name diabetes --models llata_tsne --use_3d_tsne --use_knn_connections --knn_k 5 --output_dir ./results
    
    # Customizing image size and DPI for VLM compatibility
    python evaluate_llm_baselines.py --dataset_name adult --models llata_tsne --max_vlm_image_size 1024 --image_dpi 72 --output_dir ./results
    
    # Disable RGB conversion if needed (keeping RGBA mode)
    python evaluate_llm_baselines.py --dataset_name diabetes --models llata_tsne --no-force_rgb_mode --output_dir ./results
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
import glob
import datetime
import random
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union

from clam.data import load_datasets
from sklearn.model_selection import train_test_split
from clam.utils import setup_logging, timeout_context, MetricsLogger

# Import wandb conditionally to avoid dependency issues if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import GPU monitoring and JSON utilities
from clam.utils import (
    init_wandb_with_gpu_monitoring, 
    cleanup_gpu_monitoring, 
    GPUMonitor,
    safe_json_dump, 
    convert_for_json_serialization, 
    save_results
)

# Import LLM baseline evaluation functions
from llm_baselines.tabllm_baseline import evaluate_tabllm
from llm_baselines.tabula_8b_baseline import evaluate_tabula_8b
from llm_baselines.jolt_baseline import evaluate_jolt
from llm_baselines.llata_tsne_baseline import evaluate_llata_tsne

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM baselines (TabLLM, Tabula-8B, JOLT, LlaTa-T-SNe) on tabular datasets")
    
    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        default="tabllm,tabula_8b,jolt,llata_tsne",
        help="Comma-separated list of models to evaluate: 'tabllm', 'tabula_8b', 'jolt', 'llata_tsne'"
    )
    
    # Dataset source options (mutually exclusive)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset_name",
        type=str,
        help="Name of a single dataset to evaluate on"
    )
    dataset_group.add_argument(
        "--dataset_ids",
        type=str,
        help="Comma-separated list of OpenML dataset IDs to evaluate on"
    )
    dataset_group.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing CSV files to evaluate on"
    )
    dataset_group.add_argument(
        "--num_datasets",
        type=int,
        help="Number of random datasets to sample from OpenML for evaluation"
    )
    
    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results"
    )
    
    # Model-specific configurations
    parser.add_argument(
        "--tabllm_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name for TabLLM baseline"
    )
    parser.add_argument(
        "--tabula_model",
        type=str,
        default="mlfoundations/tabula-8b",
        help="HuggingFace model name for Tabula-8B baseline"
    )
    parser.add_argument(
        "--jolt_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name for JOLT baseline (should be a generative LLM)"
    )
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
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache TabPFN embeddings for LlaTa-T-SNe baseline"
    )
    parser.add_argument(
        "--force_recompute_embeddings",
        action="store_true",
        help="Force recomputation of cached embeddings for LlaTa-T-SNe baseline"
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
    parser.add_argument(
        "--tsne_zoom_factor",
        type=float,
        default=4.0,
        help="Zoom factor for t-SNE visualizations (2.0 = 200%% zoom, showing 50%% of the range) (LlaTa-T-SNe baseline)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--k_shot",
        type=int,
        default=None,
        help="Number of training examples per class for few-shot learning (for dataset splitting). If None, uses full training set."
    )
    parser.add_argument(
        "--num_few_shot_examples",
        type=int,
        default=32,
        help="Number of few-shot examples to use for in-context learning"
    )
    parser.add_argument(
        "--balanced_few_shot",
        action="store_true",
        help="Use class-balanced few-shot examples in LLM prompts instead of random selection"
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=4096,
        help="Maximum context length for LLM models"
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Maximum number of test samples to evaluate (for debugging)"
    )
    parser.add_argument(
        "--feature_selection_threshold",
        type=int,
        default=500,
        help="Apply feature selection if dataset has more than this many features"
    )
    
    # Hardware and performance
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for evaluation"
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="GPU index to use when device is cuda"
    )
    parser.add_argument(
        "--timeout_minutes",
        type=int,
        default=30,
        help="Timeout for each model evaluation in minutes"
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
    
    # Weights & Biases logging
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="llm-baselines",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity name"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    
    # VLM Configuration
    parser.add_argument(
        "--use_semantic_names",
        action="store_true",
        help="Use semantic class names in prompts instead of 'Class X' format"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()




def apply_k_shot_split(dataset: Dict, k_shot: int, random_state: int = 42) -> Dict:
    """
    Apply k-shot splitting to a dataset.
    
    Args:
        dataset: Dataset dictionary with X, y, etc.
        k_shot: Number of samples per class
        random_state: Random seed for reproducibility
        
    Returns:
        Modified dataset with k-shot training split
    """
    logger = logging.getLogger(__name__)
    
    X = dataset["X"]
    y = dataset["y"]
    
    # Convert to numpy arrays if needed
    if hasattr(X, 'values'):
        X_array = X.values
        original_columns = X.columns.tolist()
    else:
        X_array = np.array(X)
        original_columns = dataset.get("attribute_names", [f"feature_{i}" for i in range(X_array.shape[1])])
    
    if hasattr(y, 'values'):
        y_array = y.values
    else:
        y_array = np.array(y)
    
    # Create k-shot training set (class-balanced)
    X_kshot = []
    y_kshot = []
    
    unique_classes = np.unique(y_array)
    logger.info(f"Applying k-shot split: {k_shot} samples per class, {len(unique_classes)} classes")
    
    for class_label in unique_classes:
        class_mask = y_array == class_label
        class_X = X_array[class_mask]
        class_y = y_array[class_mask]
        
        # Select k samples per class
        n_samples = min(k_shot, len(class_X))
        if n_samples < k_shot:
            logger.warning(f"Class {class_label} only has {n_samples} samples, requested {k_shot}")
        
        selected_idx = np.random.RandomState(random_state).choice(
            len(class_X), n_samples, replace=False
        )
        
        X_kshot.append(class_X[selected_idx])
        y_kshot.append(class_y[selected_idx])
    
    # Combine all selected samples
    X_kshot = np.vstack(X_kshot)
    y_kshot = np.concatenate(y_kshot)
    
    # Convert back to original format
    if hasattr(dataset["X"], 'iloc'):  # DataFrame
        import pandas as pd
        X_kshot = pd.DataFrame(X_kshot, columns=original_columns)
    
    if hasattr(dataset["y"], 'iloc'):  # Series
        import pandas as pd
        y_kshot = pd.Series(y_kshot)
    
    # Create new dataset with k-shot training data
    dataset_kshot = dataset.copy()
    dataset_kshot["X"] = X_kshot
    dataset_kshot["y"] = y_kshot
    
    logger.info(f"K-shot dataset: {len(X_kshot)} total samples ({k_shot} per class * {len(unique_classes)} classes)")
    
    return dataset_kshot


def apply_balanced_few_shot_selection(X_train, y_train, num_examples: int, random_state: int = 42):
    """
    Select few-shot examples with class balance.
    
    Args:
        X_train: Training features
        y_train: Training labels  
        num_examples: Total number of examples to select
        random_state: Random seed
        
    Returns:
        Tuple of (selected_indices, actual_num_selected)
    """
    unique_classes = np.unique(y_train)
    n_classes = len(unique_classes)
    
    # Calculate examples per class (as evenly as possible)
    examples_per_class = num_examples // n_classes
    remainder = num_examples % n_classes
    
    selected_indices = []
    
    for i, class_label in enumerate(unique_classes):
        class_mask = y_train == class_label
        class_indices = np.where(class_mask)[0]
        
        # Add one extra example to first 'remainder' classes
        n_select = examples_per_class + (1 if i < remainder else 0)
        n_select = min(n_select, len(class_indices))
        
        if n_select > 0:
            selected_class_indices = np.random.RandomState(random_state).choice(
                class_indices, n_select, replace=False
            )
            selected_indices.extend(selected_class_indices)
    
    return np.array(selected_indices), len(selected_indices)


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"llm_baseline_evaluation_{timestamp}.log"
    logger = setup_logging(log_file=os.path.join(args.output_dir, log_filename))
    logger.info(f"Arguments: {args}")
    
    # Initialize Weights & Biases with GPU monitoring if requested
    gpu_monitor = None
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("Weights & Biases requested but not installed. Run 'pip install wandb' to install.")
        else:
            if args.wandb_name is None:
                args.wandb_name = f"llm_baselines_{timestamp}"
            
            gpu_monitor = init_wandb_with_gpu_monitoring(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config=vars(args),
                output_dir=args.output_dir,
                enable_system_monitoring=True,
                gpu_log_interval=30.0,
                enable_detailed_gpu_logging=True
            )
            logger.info(f"Initialized Weights & Biases run with GPU monitoring: {args.wandb_name}")
    
    # Load datasets
    datasets = load_datasets(args)
    if not datasets:
        logger.error("No datasets loaded successfully. Exiting.")
        return
    
    # Apply k-shot splitting if requested
    if args.k_shot is not None:
        logger.info(f"Applying k-shot splitting: {args.k_shot} samples per class")
        datasets = [apply_k_shot_split(dataset, args.k_shot, args.seed) for dataset in datasets]
    
    # Parse models to evaluate
    models_to_evaluate = [model.strip() for model in args.models.split(',')]
    logger.info(f"Evaluating models: {models_to_evaluate}")
    
    # Evaluate each model on each dataset
    all_results = []
    
    for dataset in datasets:
        logger.info(f"\\n{'='*50}\\nEvaluating dataset: {dataset['name']}\\n{'='*50}")
        
        dataset_results = []
        
        for model_name in models_to_evaluate:
            logger.info(f"Evaluating {model_name} on {dataset['name']}")
            
            try:
                # Set timeout for each model evaluation
                timeout_seconds = args.timeout_minutes * 60
                
                with timeout_context(timeout_seconds):
                    if model_name.lower() == 'tabllm':
                        result = evaluate_tabllm(dataset, args)
                    elif model_name.lower() == 'tabula_8b':
                        result = evaluate_tabula_8b(dataset, args)
                    elif model_name.lower() == 'jolt':
                        result = evaluate_jolt(dataset, args)
                    elif model_name.lower() == 'llata_tsne':
                        result = evaluate_llata_tsne(dataset, args)
                    else:
                        logger.warning(f"Unknown model: {model_name}. Skipping.")
                        continue
                
            except TimeoutError:
                logger.warning(f"Evaluation of {model_name} on {dataset['name']} timed out after {args.timeout_minutes} minutes")
                result = {
                    'model_name': model_name,
                    'dataset_name': dataset['name'],
                    'dataset_id': dataset['id'],
                    'error': f'Timeout after {args.timeout_minutes} minutes',
                    'timeout': True
                }
            except Exception as e:
                logger.error(f"Error evaluating {model_name} on {dataset['name']}: {e}")
                result = {
                    'model_name': model_name,
                    'dataset_name': dataset['name'],
                    'dataset_id': dataset['id'],
                    'error': str(e)
                }
            
            dataset_results.append(result)
            all_results.append(result)
            
            # Log to wandb using unified metrics system
            if args.use_wandb and WANDB_AVAILABLE:
                # Initialize unified metrics logger for this model/dataset pair
                metrics_logger = MetricsLogger(
                    model_name=model_name,
                    dataset_name=dataset['name'],
                    use_wandb=True,
                    logger=logger
                )
                
                # Add explicit dataset_id to the result before logging
                result_with_dataset_id = result.copy()
                result_with_dataset_id['dataset_id'] = dataset['id']
                result_with_dataset_id['task_id'] = dataset['id']  # For LLATA compatibility, task_id = dataset_id
                
                # Log all metrics using unified system
                metrics_logger.log_all_metrics(result_with_dataset_id)
        
        # Save results for this dataset
        save_results(dataset_results, args.output_dir, dataset['name'])
    
    # Save aggregated results using robust JSON serialization
    aggregated_file = os.path.join(args.output_dir, "aggregated_results.json")
    success = safe_json_dump(
        all_results, 
        aggregated_file, 
        logger=logger,
        minimal_fallback=True,
        indent=2
    )
    
    if success:
        print(f"Successfully saved aggregated results to {aggregated_file}")
    else:
        logger.error(f"Failed to save aggregated results to {aggregated_file}")
    
    # Log model-level aggregation metrics using unified system
    if args.use_wandb and WANDB_AVAILABLE:
        for model_name in models_to_evaluate:
            model_results = [r for r in all_results if r.get('model_name') == model_name and 'accuracy' in r and not r.get('timeout', False)]
            if model_results:
                # Initialize aggregation metrics logger
                agg_metrics_logger = MetricsLogger(
                    model_name=model_name,
                    dataset_name="aggregated",  # Special dataset name for aggregated metrics
                    use_wandb=True,
                    logger=logger
                )
                
                # Log aggregated metrics
                agg_metrics_logger.log_aggregated_metrics(model_results, prefix="average")
                
                # Log number of valid datasets as a special metric
                agg_metrics_logger._log_metric("num_valid_datasets", len(model_results))
    
    # Print summary
    logger.info(f"\\n{'='*50}\\nEVALUATION SUMMARY\\n{'='*50}")
    
    for model_name in models_to_evaluate:
        model_results = [r for r in all_results if r.get('model_name') == model_name]
        if model_results:
            accuracies = [r['accuracy'] for r in model_results if 'accuracy' in r]
            completion_rates = [r.get('completion_rate', 1.0) for r in model_results]
            timeouts = len([r for r in model_results if r.get('timeout', False)])
            errors = len([r for r in model_results if 'error' in r and not r.get('timeout', False)])
            
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                avg_completion = np.mean(completion_rates)
                logger.info(f"{model_name}: Average accuracy = {avg_accuracy:.4f}, completion rate = {avg_completion:.1%} ({len(accuracies)} datasets)")
                if timeouts > 0:
                    logger.info(f"  - {timeouts} timeouts")
                if errors > 0:
                    logger.info(f"  - {errors} errors")
    
    logger.info(f"\\nResults saved to: {args.output_dir}")
    logger.info(f"Aggregated results: {aggregated_file}")
    
    # Clean up GPU monitoring
    cleanup_gpu_monitoring(gpu_monitor)

if __name__ == "__main__":
    main()