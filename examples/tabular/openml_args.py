#!/usr/bin/env python
"""
Shared argument configuration for OpenML evaluation scripts.

This module provides consistent argument parsing and default configuration
for OpenML collection evaluation scripts.
"""

import argparse
import sys
import os
from typing import List, Dict, Any

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from clam.utils.evaluation_args import create_tabular_llm_evaluation_parser


def create_openml_collection_parser(
    description: str,
    collection_name: str,
    task_type: str = "classification",
    default_models: List[str] = None
) -> argparse.ArgumentParser:
    """
    Create an argument parser for OpenML collection evaluation scripts.
    
    Args:
        description: Description for the argument parser
        collection_name: Name of the collection (e.g., "CC18", "regression_2025")
        task_type: Type of tasks in collection ("classification" or "regression")
        default_models: Default models to evaluate
        
    Returns:
        Configured argument parser
    """
    parser = create_tabular_llm_evaluation_parser(description)
    
    # Remove the dataset source requirement since we automatically fetch from collection
    for action_group in parser._mutually_exclusive_groups:
        if any(action.dest in ['dataset_name', 'dataset_ids', 'data_dir', 'num_datasets'] 
               for action in action_group._group_actions):
            action_group.required = False
            break
    
    # Add collection-specific orchestration arguments
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
        help=f"Number of different train/test splits to use for each task in {collection_name}"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help=f"Start from this task index in the {collection_name} collection"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help=f"End at this task index in the {collection_name} collection (exclusive)"
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
    parser.add_argument(
        "--validate_metadata_only",
        action="store_true",
        help="Only validate metadata coverage, don't run evaluations"
    )
    parser.add_argument(
        "--skip_missing_metadata",
        action="store_true",
        help="Automatically skip tasks with incomplete metadata instead of failing"
    )
    
    return parser


def configure_openml_defaults(
    parser: argparse.ArgumentParser,
    collection_name: str,
    task_type: str = "classification",
    default_models: List[str] = None,
    wandb_project_suffix: str = None
) -> None:
    """
    Configure defaults for an OpenML collection parser.
    
    Args:
        parser: The argument parser to configure
        collection_name: Name of the collection (e.g., "CC18", "regression_2025")
        task_type: Type of tasks in collection ("classification" or "regression")
        default_models: Default models to evaluate
        wandb_project_suffix: Suffix for wandb project name
    """
    if default_models is None:
        if task_type == "regression":
            default_models = ["clam_tsne", "tabllm", "jolt"]  # Remove tabula_8b for regression
        else:
            default_models = ["tabllm", "tabula_8b", "jolt", "clam_tsne"]
    
    if wandb_project_suffix is None:
        wandb_project_suffix = f"llm-baselines-{collection_name.lower()}"
    
    # Set collection-specific defaults
    defaults = {
        "output_dir": f"./openml_{collection_name.lower()}_llm_baselines_results",
        "wandb_project": f"clam-{wandb_project_suffix}",
        "models": default_models,
        "num_few_shot_examples": 16
    }
    
    # Add task-type specific defaults
    if task_type == "regression":
        defaults["preserve_regression"] = True
    
    parser.set_defaults(**defaults)


def parse_openml_collection_args(
    description: str,
    collection_name: str,
    task_type: str = "classification",
    default_models: List[str] = None,
    wandb_project_suffix: str = None
) -> argparse.Namespace:
    """
    Parse arguments for an OpenML collection evaluation script.
    
    Args:
        description: Description for the argument parser
        collection_name: Name of the collection (e.g., "CC18", "regression_2025")
        task_type: Type of tasks in collection ("classification" or "regression")
        default_models: Default models to evaluate
        wandb_project_suffix: Suffix for wandb project name
        
    Returns:
        Parsed arguments
    """
    parser = create_openml_collection_parser(description, collection_name, task_type, default_models)
    configure_openml_defaults(parser, collection_name, task_type, default_models, wandb_project_suffix)
    
    args = parser.parse_args()
    
    # Convert models back to comma-separated string for internal processing
    if isinstance(args.models, list):
        args.models = ",".join(args.models)
    
    return args


def get_cc18_args() -> argparse.Namespace:
    """Get arguments configured for OpenML CC18 collection."""
    return parse_openml_collection_args(
        description="Evaluate LLM baselines on OpenML CC18 collection",
        collection_name="CC18",
        task_type="classification",
        wandb_project_suffix="openml-cc18-llm-baselines"
    )


def get_regression_2025_args() -> argparse.Namespace:
    """Get arguments configured for OpenML regression 2025 collection."""
    return parse_openml_collection_args(
        description="Evaluate LLM baselines on New OpenML Suite 2025 regression collection",
        collection_name="regression_2025",
        task_type="regression",
        wandb_project_suffix="regression-llm-baselines-2025"
    )