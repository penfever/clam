"""
Shared argument parsing utilities for LLATA evaluation scripts.

This module provides common argument parsing functions to avoid code duplication
between evaluation scripts and make the main evaluation logic more focused.
"""

import argparse
from typing import Optional


def add_model_args(parser: argparse.ArgumentParser):
    """Add model-related arguments for evaluation."""
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the pretrained model directory (for LLATA models)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="Model identifier: 'catboost', 'tabpfn_v2', 'random_forest', 'logistic_regression', 'gradient_boosting', "
             "or a Hugging Face model ID like 'Qwen/Qwen2.5-3B-Instruct' for LLATA models. "
             "When used with --model_path, specifies the base model architecture for fallback model creation."
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=1000,
        help="Size of the embeddings (must match the pretrained model)"
    )


def add_dataset_source_args(parser: argparse.ArgumentParser):
    """Add mutually exclusive dataset source arguments."""
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the OpenML dataset to evaluate on (e.g., 'har', 'airlines', 'albert', 'volkert', 'higgs')"
    )
    dataset_group.add_argument(
        "--dataset_ids",
        type=str,
        help="Comma-separated list of OpenML dataset IDs to evaluate on"
    )
    dataset_group.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing CSV files to use as datasets"
    )
    dataset_group.add_argument(
        "--num_datasets",
        type=int,
        help="Number of random datasets to sample from OpenML"
    )


def add_data_processing_args(parser: argparse.ArgumentParser):
    """Add data processing and sampling arguments."""
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Maximum number of test samples to use for evaluation"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples to use for baseline training and TabPFN embeddings"
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        choices=["balanced", "random"],
        default="balanced",
        help="Sampling strategy when limiting training data: 'balanced' (equal samples per class) or 'random'"
    )


def add_embedding_args(parser: argparse.ArgumentParser):
    """Add embedding cache and computation arguments."""
    parser.add_argument(
        "--embedding_cache_dir",
        type=str,
        default="./data",
        help="Directory to store cached embeddings. Set to 'none' to disable caching."
    )
    parser.add_argument(
        "--force_recompute_embeddings",
        action="store_true",
        help="Force recomputation of embeddings even if cache exists"
    )


def add_baseline_model_args(parser: argparse.ArgumentParser):
    """Add baseline model hyperparameter arguments."""
    # CatBoost parameters
    parser.add_argument(
        "--catboost_iterations",
        type=int,
        default=1000,
        help="Number of iterations for CatBoost"
    )
    parser.add_argument(
        "--catboost_depth",
        type=int,
        default=6,
        help="Tree depth for CatBoost"
    )
    parser.add_argument(
        "--catboost_learning_rate", 
        type=float,
        default=0.03,
        help="Learning rate for CatBoost"
    )
    
    # Random Forest parameters
    parser.add_argument(
        "--rf_n_estimators",
        type=int,
        default=100,
        help="Number of trees in Random Forest"
    )
    parser.add_argument(
        "--rf_max_depth", 
        type=int,
        default=None,
        help="Maximum depth of trees in Random Forest"
    )
    
    # Gradient Boosting parameters
    parser.add_argument(
        "--gb_n_estimators",
        type=int,
        default=100,
        help="Number of trees in Gradient Boosting"
    )
    parser.add_argument(
        "--gb_learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for Gradient Boosting"
    )
    
    # Logistic Regression parameters
    parser.add_argument(
        "--lr_max_iter",
        type=int,
        default=1000,
        help="Maximum number of iterations for Logistic Regression"
    )
    parser.add_argument(
        "--lr_C",
        type=float,
        default=1.0,
        help="Regularization strength for Logistic Regression (smaller values = stronger regularization)"
    )


def add_tabpfn_args(parser: argparse.ArgumentParser):
    """Add TabPFN v2 specific arguments."""
    parser.add_argument(
        "--tabpfn_v2_path",
        type=str,
        default=None,
        help="Path to TabPFN v2 model file (if not specified, will use the default path)"
    )
    parser.add_argument(
        "--tabpfn_v2_N_ensemble_configurations",
        type=int,
        default=32,
        help="Number of ensemble configurations for TabPFN v2"
    )


def add_evaluation_wandb_args(parser: argparse.ArgumentParser, default_project: str = "llata-evaluation"):
    """Add Weights & Biases arguments for evaluation."""
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=default_project,
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity (team) name"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name (defaults to 'eval_dataset' + timestamp)"
    )


def add_label_fitting_args(parser: argparse.ArgumentParser):
    """Add label fitting and adjustment arguments."""
    parser.add_argument(
        "--label_fitting", 
        action="store_true", 
        help="Adjust predicted labels to match dataset label frequency distribution using optimal permutation"
    )
    parser.add_argument(
        "--label_fitting_threshold", 
        type=float, 
        default=0.05,
        help="Frequency difference threshold for label remapping (default: 0.05)"
    )
    parser.add_argument(
        "--label_fitting_holdout_ratio", 
        type=float, 
        default=0.1,
        help="Ratio of training set to use for label fitting (default: 0.1)"
    )


def add_calibration_args(parser: argparse.ArgumentParser):
    """Add probability calibration arguments."""
    parser.add_argument(
        "--baseline_calibration",
        action="store_true",
        help="Use baseline probability calibration to adjust for model's prior label token biases"
    )
    parser.add_argument(
        "--baseline_sample_ratio",
        type=float,
        default=0.1,
        help="Ratio of training set to use for computing baseline probabilities (default: 0.1)"
    )


def add_score_normalization_args(parser: argparse.ArgumentParser):
    """Add score normalization arguments."""
    parser.add_argument(
        "--score_normalization",
        type=str,
        choices=["none", "temperature", "isotonic", "histogram"],
        default="none",
        help="Type of score normalization to apply (default: none)"
    )
    parser.add_argument(
        "--normalization_temperature",
        type=float,
        default=2.0,
        help="Temperature for temperature-based normalization (default: 2.0)"
    )


def add_minority_class_args(parser: argparse.ArgumentParser):
    """Add minority class boosting arguments."""
    parser.add_argument(
        "--minority_class_boost",
        action="store_true",
        help="Boost probabilities for minority classes to reduce bias toward frequent classes"
    )
    parser.add_argument(
        "--minority_boost_factor",
        type=float,
        default=2.0,
        help="Factor to boost minority class probabilities (default: 2.0)"
    )


def add_evaluation_control_args(parser: argparse.ArgumentParser):
    """Add evaluation control arguments."""
    parser.add_argument(
        "--only_ground_truth_classes",
        action="store_true",
        help="Only consider classes that appear in the ground truth test data (default: True)"
    )
    parser.add_argument(
        "--run_all_baselines",
        action="store_true",
        help="Run all baseline models in addition to the specified model"
    )
    parser.add_argument(
        "--baselines_only",
        action="store_true",
        help="Only run baseline models (skip LLATA evaluation)"
    )


def add_common_evaluation_args(parser: argparse.ArgumentParser):
    """Add common evaluation arguments."""
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for evaluation (auto, cuda, cpu)"
    )


def create_evaluation_parser(description: str, default_wandb_project: str = "llata-evaluation") -> argparse.ArgumentParser:
    """
    Create an argument parser with all common arguments for LLATA evaluation.
    
    Args:
        description: Description for the argument parser
        default_wandb_project: Default project name for W&B
        
    Returns:
        ArgumentParser with all common evaluation arguments
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Add all common argument groups
    add_common_evaluation_args(parser)
    add_model_args(parser)
    add_dataset_source_args(parser)
    add_data_processing_args(parser)
    add_embedding_args(parser)
    add_baseline_model_args(parser)
    add_tabpfn_args(parser)
    add_evaluation_wandb_args(parser, default_wandb_project)
    add_label_fitting_args(parser)
    add_calibration_args(parser)
    add_score_normalization_args(parser)
    add_minority_class_args(parser)
    add_evaluation_control_args(parser)
    
    return parser


def create_dataset_evaluation_parser() -> argparse.ArgumentParser:
    """Create argument parser specific to dataset evaluation."""
    parser = create_evaluation_parser(
        "Evaluate a pretrained LLATA model or baseline ML models on tabular datasets",
        default_wandb_project="llata-evaluation"
    )
    
    return parser