#!/usr/bin/env python
"""
Script to evaluate LLM baselines (TabLLM, Tabula-8B, JOLT, and CLAM-T-SNe) on the New OpenML Suite 2025 regression collection.

This script uses the shared OpenML orchestration framework for consistency and maintainability.

Usage:
    # Basic usage with all models
    python run_openml_regression_2025_llm_baselines_tabular.py --clam_repo_path /path/to/clam --output_dir ./results
    
    # Run only CLAM-T-SNe with 3D t-SNE and KNN connections
    python run_openml_regression_2025_llm_baselines_tabular.py --clam_repo_path /path/to/clam --output_dir ./results \
        --models clam_tsne --use_3d --use_knn_connections --nn_k 7
    
    # Run with custom image settings for regression visualization
    python run_openml_regression_2025_llm_baselines_tabular.py --clam_repo_path /path/to/clam --output_dir ./results \
        --models clam_tsne --max_vlm_image_size 1024 --image_dpi 72 --no-force_rgb_mode
"""

import os
import logging
import random
import numpy as np
import torch
import sys
from datetime import datetime

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import shared framework
from examples.tabular.openml_orchestration import (
    OpenMLEvaluationOrchestrator, 
    handle_metadata_validation, 
    generate_metadata_report
)
from examples.tabular.openml_collections import get_openml_regression_2025_tasks
from examples.tabular.openml_args import get_regression_2025_args

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


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # Parse arguments using shared framework
    args = get_regression_2025_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Get OpenML regression 2025 tasks using shared collection fetcher
    tasks = get_openml_regression_2025_tasks()
    
    if not tasks:
        logger.error("No regression tasks found. Exiting.")
        return
    
    # Generate metadata report using shared function
    generate_metadata_report(tasks, args, task_type="regression")
    
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
    
    # Parse models to check
    models_to_check = [model.strip() for model in args.models.split(',')]
    
    # Handle metadata validation using shared framework
    tasks = handle_metadata_validation(args, tasks, models_to_check, task_type="regression")
    if not tasks:  # Empty list means exit (validation only or no valid tasks)
        return
    
    # Create orchestrator for regression tasks
    orchestrator = OpenMLEvaluationOrchestrator(args, task_type="regression")
    
    # Process each task using shared orchestrator
    for i, task in enumerate(tasks):
        try:
            logger.info(f"Processing regression task {i+1}/{len(tasks)}")
            orchestrator.process_task(task)
        except Exception as e:
            logger.error(f"Error processing regression task {task.task_id}: {e}")
    
    # Aggregate results using shared framework
    try:
        orchestrator.aggregate_results()
    except Exception as e:
        logger.error(f"Error aggregating regression results: {e}")
    
    logger.info("All regression tasks completed")


if __name__ == "__main__":
    main()