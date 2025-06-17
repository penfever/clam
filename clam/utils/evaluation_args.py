"""
Shared argument parsing utilities for CLAM evaluation scripts.

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
        help="Path to the pretrained model directory (for CLAM models)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="Model identifier: 'catboost', 'tabpfn_v2', 'random_forest', 'logistic_regression', 'gradient_boosting', "
             "or a Hugging Face model ID like 'Qwen/Qwen2.5-3B-Instruct' for CLAM models. "
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


def add_evaluation_wandb_args(parser: argparse.ArgumentParser, default_project: str = "clam-evaluation"):
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
        help="Only run baseline models (skip CLAM evaluation)"
    )
    parser.add_argument(
        "--use_semantic_names",
        action="store_true",
        help="Use semantic class names in prompts instead of 'Class X' format"
    )


def add_audio_args(parser: argparse.ArgumentParser):
    """Add audio-specific arguments."""
    parser.add_argument(
        "--embedding_model",
        type=str,
        choices=["whisper", "clap"],
        default="whisper",
        help="Audio embedding model to use"
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        choices=["tiny", "base", "small", "medium", "large", "large-v2"],
        default="large-v2",
        help="Whisper model variant to use"
    )
    parser.add_argument(
        "--clap_version",
        type=str,
        choices=["2022", "2023", "clapcap"],
        default="2023",
        help="CLAP model version to use"
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=5,
        help="Number of training examples per class for few-shot learning"
    )
    parser.add_argument(
        "--audio_duration",
        type=float,
        default=None,
        help="Maximum audio duration to process (seconds)"
    )
    parser.add_argument(
        "--num_few_shot_examples",
        type=int,
        default=32,
        help="Number of few-shot examples for baseline evaluation"
    )
    parser.add_argument(
        "--balanced_few_shot",
        action="store_true",
        help="Use balanced sampling for few-shot examples"
    )
    parser.add_argument(
        "--include_spectrogram",
        action="store_true",
        default=True,
        help="Include spectrogram visualization in CLAM t-SNE"
    )
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="Do not download datasets (assume they exist locally)"
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run quick test with subset of data"
    )
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        default=True,
        help="Save visualizations and VLM responses"
    )
    parser.add_argument(
        "--no_save_outputs",
        dest="save_outputs",
        action="store_false",
        help="Disable saving visualizations and VLM responses"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["clam_tsne"],
        choices=["clam_tsne", "whisper_knn", "clap_zero_shot"],
        help="List of models to evaluate (default: clam_tsne)"
    )


def add_vision_args(parser: argparse.ArgumentParser):
    """Add vision-specific arguments."""
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to custom dataset (required for custom datasets)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        help="Number of classes (required for custom datasets)"
    )
    parser.add_argument(
        "--imagenet_subset",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="ImageNet subset to use"
    )
    parser.add_argument(
        "--dinov2_model",
        type=str,
        choices=[
            "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
            "dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg"
        ],
        default="dinov2_vits14",
        help="DINOv2 model variant to use"
    )
    parser.add_argument(
        "--max_train_plot_samples",
        type=int,
        default=1000,
        help="Maximum number of training samples to show in t-SNE plots"
    )
    parser.add_argument(
        "--knn_neighbors",
        type=int,
        default=5,
        help="Number of neighbors for KNN classifier"
    )
    parser.add_argument(
        "--vlm_model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Vision Language Model ID for CLAM t-SNE evaluation"
    )
    parser.add_argument(
        "--bioclip2_model",
        type=str,
        default="hf-hub:imageomics/bioclip-2",
        help="BioClip2 model identifier for biological datasets"
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run quick test with subset of data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./vision_data",
        help="Base directory for vision datasets"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["clam_tsne", "dinov2_linear"],
        choices=["clam_tsne", "clam_simple", "dinov2_linear", "qwen_vl"],
        help="List of models to evaluate (default: clam_tsne and dinov2_linear)"
    )


def add_tsne_visualization_args(parser: argparse.ArgumentParser):
    """Add t-SNE and visualization arguments."""
    parser.add_argument(
        "--use_3d_tsne",
        action="store_true",
        help="Use 3D t-SNE visualization with multiple viewing angles"
    )
    parser.add_argument(
        "--use_knn_connections",
        action="store_true",
        help="Show KNN connections in t-SNE visualization"
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=5,
        help="Number of nearest neighbors for KNN connections/analysis"
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=int,
        default=30,
        help="Perplexity parameter for t-SNE"
    )
    parser.add_argument(
        "--tsne_n_iter",
        type=int,
        default=1000,
        help="Number of iterations for t-SNE"
    )
    parser.add_argument(
        "--tsne_zoom_factor",
        type=float,
        default=4.0,
        help="Zoom factor for t-SNE visualization"
    )
    parser.add_argument(
        "--use_pca_backend",
        action="store_true",
        help="Use PCA as preprocessing backend for t-SNE"
    )
    parser.add_argument(
        "--save_every_n",
        type=int,
        default=10,
        help="Save visualization every N samples (for debugging)"
    )
    parser.add_argument(
        "--max_vlm_image_size",
        type=int,
        default=768,
        help="Maximum image size for VLM processing"
    )
    parser.add_argument(
        "--image_dpi",
        type=int,
        default=100,
        help="DPI for generated visualizations"
    )
    parser.add_argument(
        "--force_rgb_mode",
        action="store_true",
        help="Force RGB mode for generated images"
    )
    parser.add_argument(
        "--no-force_rgb_mode",
        action="store_false",
        dest="force_rgb_mode",
        help="Allow RGBA mode for generated images"
    )
    parser.add_argument(
        "--save_sample_visualizations",
        action="store_true",
        default=True,
        help="Save sample t-SNE visualizations for debugging"
    )
    parser.add_argument(
        "--no-save_sample_visualizations",
        action="store_false",
        dest="save_sample_visualizations",
        help="Disable saving of sample t-SNE visualizations"
    )
    parser.add_argument(
        "--viewing_angles",
        type=str,
        help="Custom viewing angles for 3D t-SNE (format: 'elev1,azim1;elev2,azim2;...')"
    )


def add_llm_baseline_args(parser: argparse.ArgumentParser):
    """Add LLM baseline model arguments."""
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tabllm", "tabula_8b", "jolt", "clam_tsne"],
        choices=["tabllm", "tabula_8b", "jolt", "clam_tsne"],
        help="List of models to evaluate (default: all models)"
    )
    parser.add_argument(
        "--tabllm_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="TabLLM model identifier"
    )
    parser.add_argument(
        "--tabula_model",
        type=str,
        default="mlfoundations/tabula-8b",
        help="Tabula-8B model identifier"
    )
    parser.add_argument(
        "--jolt_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="JOLT baseline model identifier"
    )
    parser.add_argument(
        "--max_tabpfn_samples",
        type=int,
        default=3000,
        help="Maximum number of samples for TabPFN embedding computation"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        help="Backend for LLM inference (vllm, transformers, etc.)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for distributed inference"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization factor"
    )
    parser.add_argument(
        "--timeout_minutes",
        type=int,
        default=30,
        help="Timeout for each model evaluation in minutes"
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=None,
        help="Number of training examples per class for few-shot learning"
    )


def add_dataset_selection_args(parser: argparse.ArgumentParser):
    """Add dataset selection arguments for different modalities."""
    parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of datasets to evaluate on"
    )
    parser.add_argument(
        "--audio_datasets",
        nargs="+",
        choices=["esc50", "ravdess", "urbansound8k"],
        help="Audio datasets to evaluate on"
    )
    parser.add_argument(
        "--vision_datasets",
        nargs="+",
        choices=["cifar10", "cifar100", "imagenet", "custom"],
        help="Vision datasets to evaluate on"
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


def create_evaluation_parser(description: str, default_wandb_project: str = "clam-evaluation") -> argparse.ArgumentParser:
    """
    Create an argument parser with all common arguments for CLAM evaluation.
    
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
        "Evaluate a pretrained CLAM model or baseline ML models on tabular datasets",
        default_wandb_project="clam-evaluation"
    )
    
    return parser


def create_audio_evaluation_parser(description: str = "Evaluate CLAM and baseline models on audio datasets") -> argparse.ArgumentParser:
    """Create argument parser for audio evaluation with all relevant argument groups."""
    parser = argparse.ArgumentParser(description=description)
    
    # Add common evaluation arguments
    add_common_evaluation_args(parser)
    add_evaluation_wandb_args(parser, "clam-audio-evaluation")
    add_evaluation_control_args(parser)
    
    # Add audio-specific arguments
    add_audio_args(parser)
    add_dataset_selection_args(parser)
    add_tsne_visualization_args(parser)
    
    return parser


def create_vision_evaluation_parser(description: str = "Evaluate CLAM and baseline models on vision datasets") -> argparse.ArgumentParser:
    """Create argument parser for vision evaluation with all relevant argument groups."""
    parser = argparse.ArgumentParser(description=description)
    
    # Add common evaluation arguments
    add_common_evaluation_args(parser)
    add_evaluation_wandb_args(parser, "clam-vision-evaluation")
    add_evaluation_control_args(parser)
    
    # Add vision-specific arguments
    add_vision_args(parser)
    add_dataset_selection_args(parser)
    add_tsne_visualization_args(parser)
    
    return parser


def create_tabular_llm_evaluation_parser(description: str = "Evaluate LLM baselines on tabular datasets") -> argparse.ArgumentParser:
    """Create argument parser for tabular LLM baseline evaluation."""
    parser = argparse.ArgumentParser(description=description)
    
    # Add common evaluation arguments
    add_common_evaluation_args(parser)
    add_evaluation_wandb_args(parser, "clam-tabular-llm-evaluation")
    add_evaluation_control_args(parser)
    
    # Add dataset arguments
    add_dataset_source_args(parser)
    add_data_processing_args(parser)
    
    # Add LLM and t-SNE specific arguments
    add_llm_baseline_args(parser)
    add_tsne_visualization_args(parser)
    
    return parser


def create_multimodal_evaluation_parser(description: str = "Evaluate CLAM models across multiple modalities") -> argparse.ArgumentParser:
    """Create argument parser with all modality-specific arguments."""
    parser = argparse.ArgumentParser(description=description)
    
    # Add all argument groups for maximum flexibility
    add_common_evaluation_args(parser)
    add_model_args(parser)
    add_dataset_source_args(parser)
    add_data_processing_args(parser)
    add_embedding_args(parser)
    add_baseline_model_args(parser)
    add_tabpfn_args(parser)
    add_evaluation_wandb_args(parser, "clam-multimodal-evaluation")
    add_label_fitting_args(parser)
    add_calibration_args(parser)
    add_score_normalization_args(parser)
    add_minority_class_args(parser)
    add_evaluation_control_args(parser)
    
    # Add modality-specific arguments
    add_audio_args(parser)
    add_vision_args(parser)
    add_dataset_selection_args(parser)
    add_tsne_visualization_args(parser)
    add_llm_baseline_args(parser)
    
    return parser