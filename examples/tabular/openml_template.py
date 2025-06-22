#!/usr/bin/env python
"""
Template for creating new OpenML collection evaluation scripts.

Copy this file and customize it for your specific OpenML collection.

Usage:
    1. Copy this file to your collection directory
    2. Update COLLECTION_NAME, STUDY_ID, and TASK_TYPE
    3. Customize any collection-specific logic
    4. Run the script
"""

import os
import logging
import random
import numpy as np
import torch
import sys
from datetime import datetime

# ========================================
# CUSTOMIZE THESE SETTINGS FOR YOUR COLLECTION
# ========================================

COLLECTION_NAME = "your_collection_name"  # e.g., "CC18", "regression_2025"
STUDY_ID = None  # OpenML study ID, e.g., 99 for CC18, 455 for regression_2025
TASK_TYPE = "classification"  # "classification" or "regression"
DEFAULT_MODELS = ["tabllm", "jolt", "clam_tsne"]  # Default models to evaluate

# ========================================

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import shared framework
from examples.tabular.openml_orchestration import (
    OpenMLEvaluationOrchestrator, 
    handle_metadata_validation, 
    generate_metadata_report
)
from examples.tabular.openml_collections import get_openml_collection_tasks
from examples.tabular.openml_args import parse_openml_collection_args

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"openml_{COLLECTION_NAME.lower()}_llm_baselines.log"),
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


def get_collection_tasks():
    """
    Get tasks for this collection.
    
    Customize this function if you need special logic for fetching tasks.
    """
    if STUDY_ID is not None:
        return get_openml_collection_tasks(COLLECTION_NAME, STUDY_ID)
    else:
        # If no study ID, you'll need to implement custom task fetching
        # or use get_openml_collection_tasks with a known collection name
        return get_openml_collection_tasks(COLLECTION_NAME)


def main():
    # Parse arguments using shared framework
    args = parse_openml_collection_args(
        description=f"Evaluate LLM baselines on OpenML {COLLECTION_NAME} collection",
        collection_name=COLLECTION_NAME,
        task_type=TASK_TYPE,
        default_models=DEFAULT_MODELS
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Get collection tasks
    tasks = get_collection_tasks()
    
    if not tasks:
        logger.error(f"No {COLLECTION_NAME} tasks found. Exiting.")
        return
    
    # Generate metadata report using shared function
    generate_metadata_report(tasks, args, task_type=TASK_TYPE)
    
    # Filter tasks if task_ids is provided
    if args.task_ids:
        task_ids = [int(id.strip()) for id in args.task_ids.split(",")]
        tasks = [task for task in tasks if task.task_id in task_ids]
        logger.info(f"Filtered to {len(tasks)} specified {COLLECTION_NAME} tasks")
    
    # Apply start and end indices
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(tasks)
    tasks = tasks[start_idx:end_idx]
    logger.info(f"Processing {COLLECTION_NAME} tasks from index {start_idx} to {end_idx} (total: {len(tasks)})")
    
    # Parse models to check
    models_to_check = [model.strip() for model in args.models.split(',')]
    
    # Handle metadata validation using shared framework
    tasks = handle_metadata_validation(args, tasks, models_to_check, task_type=TASK_TYPE)
    if not tasks:  # Empty list means exit (validation only or no valid tasks)
        return
    
    # Create orchestrator for tasks
    orchestrator = OpenMLEvaluationOrchestrator(args, task_type=TASK_TYPE)
    
    # Process each task using shared orchestrator
    for i, task in enumerate(tasks):
        try:
            logger.info(f"Processing {COLLECTION_NAME} task {i+1}/{len(tasks)}")
            orchestrator.process_task(task)
        except Exception as e:
            logger.error(f"Error processing {COLLECTION_NAME} task {task.task_id}: {e}")
    
    # Aggregate results using shared framework
    try:
        orchestrator.aggregate_results()
    except Exception as e:
        logger.error(f"Error aggregating {COLLECTION_NAME} results: {e}")
    
    logger.info(f"All {COLLECTION_NAME} tasks completed")


if __name__ == "__main__":
    main()