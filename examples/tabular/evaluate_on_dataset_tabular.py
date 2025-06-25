#!/usr/bin/env python
"""
Script for evaluating a pretrained CLAM model on one or more tabular datasets.
This script handles:
1. Loading and preprocessing datasets from multiple sources
2. Computing TabPFN embeddings
3. Creating test datasets with the right format
4. Evaluating the pretrained model on these datasets
5. Optional Weights & Biases logging and visualization

Datasets can be specified in multiple ways:
- Single dataset by name or ID
- List of dataset IDs
- Directory of CSV files
- Random sampling from OpenML

Usage examples:
    # Basic usage with a single dataset
    python evaluate_on_dataset.py --model_path ./models/clam_output --dataset_name har
    
    # Using limited training samples with balanced sampling (default)
    python evaluate_on_dataset.py --model_path ./models/clam_output --dataset_name har --max_train_samples 1000
    
    # Using limited training samples with random sampling
    python evaluate_on_dataset.py --model_path ./models/clam_output --dataset_name airlines --max_train_samples 500 --sampling_strategy random
    
    # Evaluating on multiple specific datasets
    python evaluate_on_dataset.py --model_path ./models/clam_output --dataset_ids 1590,40975,37,54 --output_dir ./eval_results
    
    # Evaluating on 5 randomly sampled datasets from OpenML
    python evaluate_on_dataset.py --model_path ./models/clam_output --num_datasets 5 --output_dir ./eval_results
    
    # Evaluating on all CSV files in a directory
    python evaluate_on_dataset.py --model_path ./models/clam_output --data_dir ./datasets --output_dir ./eval_results
    
    # Using Weights & Biases for experiment tracking
    python evaluate_on_dataset.py --model_path ./models/clam_output --dataset_ids 1590,40975 --use_wandb --wandb_project myproject
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
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Any, Union

from clam.data import (
    load_dataset, 
    get_tabpfn_embeddings, 
    create_llm_dataset, 
    list_available_datasets,
    is_csv_dataset,
    load_csv_dataset,
    load_dataset_with_metadata,
    find_csv_with_fallbacks
)
from clam.models import prepare_qwen_with_prefix_embedding, QwenWithPrefixEmbedding, load_pretrained_model
from clam.models.vq import prepare_qwen_with_vq_prefix_embedding, QwenWithVQPrefixEmbedding, load_vq_pretrained_model
from clam.train import evaluate_llm_on_test_set
from clam.utils import setup_logging, MetricsLogger

# Import wandb conditionally to avoid dependency issues if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import GPU monitoring utilities
from clam.utils import init_wandb_with_gpu_monitoring, cleanup_gpu_monitoring, GPUMonitor

# Import new argument parsing utilities
from clam.utils.evaluation_args import create_dataset_evaluation_parser

# Import new dataset processing utilities
from clam.data.evaluation_utils import load_datasets_for_evaluation, preprocess_datasets_for_evaluation

def parse_args():
    """Parse command line arguments using the abstracted argument parser."""
    parser = create_dataset_evaluation_parser()
    return parser.parse_args()


def load_datasets(args) -> List[Dict[str, Any]]:
    """Load datasets and preprocess them for evaluation using abstracted utilities."""
    # Load raw datasets
    raw_datasets = load_datasets_for_evaluation(args)
    
    # Preprocess datasets (train/test split, sampling, etc.)
    processed_datasets = preprocess_datasets_for_evaluation(raw_datasets, args)
    
    return processed_datasets

def preprocess_features(X: np.ndarray, categorical_indicator: List[bool], preserve_categorical: bool = False) -> np.ndarray:
    """
    Preprocess features, converting string features to numeric values 
    and handling missing values.
    
    Args:
        X: Feature matrix
        categorical_indicator: Boolean list indicating categorical features
        preserve_categorical: If True, keep categorical features as strings for CatBoost
        
    Returns:
        Processed feature matrix
    """
    import pandas as pd
    import logging
    logger = logging.getLogger(__name__)
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(X)
    
    # Process each column
    for col_idx in range(df.shape[1]):
        col = df.iloc[:, col_idx]
        is_categorical = categorical_indicator[col_idx] if col_idx < len(categorical_indicator) else False
        
        # Check if column has object/string data
        if col.dtype == 'object' or is_categorical:
            if preserve_categorical:
                # For CatBoost, keep categorical features as strings
                logger.info(f"Preserving categorical feature at column {col_idx}")
                # Just handle missing values
                col_filled = col.fillna('missing')
                # Ensure we can assign string values by converting the column dtype first
                df.iloc[:, col_idx] = df.iloc[:, col_idx].astype(object)
                df.iloc[:, col_idx] = col_filled.astype(str)
                logger.info(f"  Preserved categorical feature with {col_filled.nunique()} unique values")
            else:
                logger.info(f"Converting feature at column {col_idx} to numeric")
                
                # For categorical features, use label encoding
                try:
                    from sklearn.preprocessing import LabelEncoder
                    # Explicitly call infer_objects to handle warnings
                    col_filled = col.infer_objects(copy=False)
                    
                    # Use label encoder
                    encoder = LabelEncoder()
                    encoded_values = encoder.fit_transform(col_filled)
                    
                    # Ensure column can accept the encoded values
                    # Convert to appropriate dtype that can hold the encoded values
                    if encoded_values.max() <= 127 and encoded_values.min() >= -128:
                        target_dtype = 'int8'
                    elif encoded_values.max() <= 32767 and encoded_values.min() >= -32768:
                        target_dtype = 'int16'
                    else:
                        target_dtype = 'int32'
                    
                    # Assign with compatible dtype
                    df.iloc[:, col_idx] = df.iloc[:, col_idx].astype(target_dtype)
                    df.iloc[:, col_idx] = pd.Series(encoded_values, index=df.index, dtype=target_dtype)
                    logger.info(f"  Encoded {len(encoder.classes_)} unique categories for column {col_idx}")
                except Exception as e:
                    logger.warning(f"  Error encoding column {col_idx}: {e}")
                    # If encoding fails, replace with zeros
                    df.iloc[:, col_idx] = df.iloc[:, col_idx].astype('int32')
                    df.iloc[:, col_idx] = pd.Series(np.zeros(len(df), dtype='int32'), index=df.index)
        else:
            # For numeric features
            if preserve_categorical:
                # For CatBoost, we can keep NaN values as it handles them natively
                # Just ensure the column is numeric type
                if col.dtype == 'object':
                    # Try to convert to numeric
                    df.iloc[:, col_idx] = pd.to_numeric(col, errors='coerce')
                    logger.info(f"  Converted object column {col_idx} to numeric for CatBoost")
                # CatBoost handles NaN values, so we don't fill them
            else:
                # For other models, fill NaN values
                if col.isna().any():
                    # If more than 75% of the values are NaN, fill with zeros
                    if col.isna().mean() > 0.75:
                        fill_value = 0
                    # Otherwise, use the median
                    else:
                        fill_value = col.median() if not np.isnan(col.median()) else 0
                    
                    # Use Series constructor to ensure type compatibility
                    filled_col = col.fillna(fill_value)
                    # Ensure compatible dtype before assignment
                    if df.iloc[:, col_idx].dtype != filled_col.dtype:
                        df.iloc[:, col_idx] = df.iloc[:, col_idx].astype(filled_col.dtype)
                    df.iloc[:, col_idx] = pd.Series(filled_col, index=df.index)
                    logger.info(f"  Filled {col.isna().sum()} missing values in column {col_idx}")
    
    # Convert back to numpy array
    X_processed = df.values
    
    return X_processed

def get_token_ids_from_model(tokenizer):
    """Extract special token IDs from tokenizer."""
    prefix_start_id = tokenizer.convert_tokens_to_ids("<PREFIX_START>")
    prefix_end_id = tokenizer.convert_tokens_to_ids("<PREFIX_END>")
    class_token_ids = [tokenizer.convert_tokens_to_ids(f"<CLASS_{i}>") for i in range(10)]
    
    return prefix_start_id, prefix_end_id, class_token_ids

def load_pretrained_model(model_path, device_map="auto", embedding_size=1000, model_id=None):
    """Load the pretrained model and tokenizer.
    
    This function automatically detects if a VQ checkpoint is passed and loads it appropriately.
    It first tries to load as a VQ model, and falls back to standard model loading if that fails.
    
    Args:
        model_path: Path to the model directory or checkpoint
        device_map: Device mapping for model loading
        embedding_size: Size of embeddings
        
    Returns:
        Tuple of (model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids, is_vq)
        where is_vq indicates whether the loaded model is a VQ model
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Check if this is actually a VQ model before attempting VQ loading
    from clam.utils import find_best_checkpoint
    resolved_path = find_best_checkpoint(model_path)
    
    # Look for VQ-specific files to determine if this is a VQ model
    vector_quantizer_path = os.path.join(resolved_path, "vector_quantizer.pt")
    has_vq_files = os.path.exists(vector_quantizer_path)
    
    if has_vq_files:
        # Only try VQ loading if VQ files are actually present
        try:
            logger.info(f"Detected VQ files, loading as VQ model from {model_path}")
            model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids, is_vq = load_vq_pretrained_model(
                model_path=model_path,
                device_map=device_map,
                embedding_size=embedding_size,
                model_id=model_id
            )
            if is_vq:
                logger.info("Successfully loaded Vector-Quantized model")
            else:
                logger.info("Loaded standard (non-VQ) model using VQ loader")
            return model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids, is_vq
        except Exception as e:
            logger.warning(f"Failed to load as VQ model despite VQ files present: {e}")
    else:
        logger.info(f"No VQ files detected, skipping VQ loading for {model_path}")
        
    # Fall back to standard model loading
    logger.info(f"Loading as standard model from {model_path}")
    from clam.utils import load_pretrained_model as utils_load_pretrained_model
    
    # Use the centralized utility function
    model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = utils_load_pretrained_model(
        model_path=model_path,
        device_map=device_map,
        embedding_size=embedding_size,
        model_id=model_id
    )
    
    # Return with is_vq=False for standard models
    return model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids, False

def load_embeddings_with_limit(cache_file, max_test_samples=None):
    """Load embeddings from cache file with optional limit on test/val samples."""
    import logging
    logger = logging.getLogger(__name__)
    cache = np.load(cache_file, allow_pickle=True)
    
    # Load train embeddings fully
    train_embeddings = cache["train_embeddings"]
    y_train_sample = cache["y_train_sample"]
    
    # Get full embeddings to check sizes 
    val_embeddings_full = cache["val_embeddings"]
    test_embeddings_full = cache["test_embeddings"]
    
    if max_test_samples is None:
        # Return full embeddings if no limit
        return train_embeddings, val_embeddings_full, test_embeddings_full, y_train_sample
    
    # Handle embeddings shape - can be either (n_samples, emb_size) or (n_ensemble, n_samples, emb_size)
    if len(val_embeddings_full.shape) == 3:
        # Multi-ensemble embeddings: (n_ensemble, n_samples, emb_size)
        _, val_count, _ = val_embeddings_full.shape
        _, test_count, _ = test_embeddings_full.shape
        
        # Calculate how many samples to load
        val_to_load = min(val_count, max_test_samples)
        test_to_load = min(test_count, max_test_samples)
        
        # Extract only the needed portions
        val_embeddings = val_embeddings_full[:, :val_to_load, :]
        test_embeddings = test_embeddings_full[:, :test_to_load, :]
    else:
        # Single embeddings: (n_samples, emb_size)
        val_count = len(val_embeddings_full)
        test_count = len(test_embeddings_full)
        
        # Calculate how many samples to load
        val_to_load = min(val_count, max_test_samples)
        test_to_load = min(test_count, max_test_samples)
        
        # Extract only the needed portions
        val_embeddings = val_embeddings_full[:val_to_load]
        test_embeddings = test_embeddings_full[:test_to_load]
    
    logger.info(f"Loaded embeddings with limits - Val: {val_to_load}/{val_count}, Test: {test_to_load}/{test_count}")
    
    return train_embeddings, val_embeddings, test_embeddings, y_train_sample

def process_dataset(dataset: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Process a single dataset: split into train/val/test, compute embeddings, and create LLM datasets.
    
    Args:
        dataset: Dictionary with dataset information
        args: Command line arguments
        
    Returns:
        Processed dataset with additional fields
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Get the dataset attributes
    X = dataset["X"]
    y = dataset["y"]
    categorical_indicator = dataset.get("categorical_indicator", [False] * X.shape[1])
    
    # Store the categorical indicator for CatBoost preprocessing later
    dataset["categorical_indicator_raw"] = categorical_indicator.copy() if hasattr(categorical_indicator, 'copy') else list(categorical_indicator)
    
    # Preprocess features to convert strings to numeric
    X = preprocess_features(X, categorical_indicator)
    
    # Determine if this is a classification or regression task
    is_classification = True
    
    # Check if labels are continuous (regression) or discrete (classification)
    if isinstance(y, np.ndarray):
        # If we have string labels, it's definitely classification
        if y.dtype.kind == 'O':
            logger.info(f"Dataset {dataset['name']} has string labels. Encoding to integers.")
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
            logger.info(f"Encoded {len(encoder.classes_)} unique classes")
        # If numeric, check if it's continuous or discrete
        elif y.dtype.kind in ('i', 'u'):  # Integer type
            # It's likely classification if few unique values
            unique_vals = np.unique(y)
            logger.info(f"Dataset {dataset['name']} has {len(unique_vals)} unique integer values")
            # Keep as classification with integer labels
        elif y.dtype.kind == 'f':  # Float type
            # Check if it's actually discrete disguised as float (e.g., 1.0, 2.0, 3.0)
            unique_vals = np.unique(y)
            if len(unique_vals) <= 10 and all(float(val).is_integer() for val in unique_vals):
                # Convert to integers for classification
                logger.info(f"Dataset {dataset['name']} has {len(unique_vals)} discrete float values. Converting to integers.")
                y = y.astype(int)
            else:
                # It's truly continuous - must bin for classification tasks
                logger.info(f"Dataset {dataset['name']} has continuous target. Converting to classification by binning.")
                # For TabPFN which expects classification, bin the continuous values into discrete categories
                from sklearn.preprocessing import KBinsDiscretizer
                # Use quantile binning to create balanced classes
                n_bins = min(10, len(np.unique(y)))  # Use at most 10 bins
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
                y = discretizer.fit_transform(y.reshape(-1, 1)).flatten().astype(int)
                logger.info(f"Binned continuous target into {n_bins} classes")
                dataset["target_discretizer"] = discretizer  # Store for future reference
    
    # Use args.seed and dataset ID to create a dataset-specific but reproducible random state
    # This ensures different datasets get different splits, but the same dataset always gets the same split
    # Use hashlib for stable hashing across Python sessions
    import hashlib
    dataset_id_bytes = str(dataset["id"]).encode('utf-8')
    dataset_id_hash = int(hashlib.md5(dataset_id_bytes).hexdigest()[:8], 16) % 10000
    dataset_specific_seed = args.seed + dataset_id_hash
    
    # Split into train, validation, and test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=dataset_specific_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=dataset_specific_seed
    )
    
    logger.info(f"Dataset {dataset['name']} shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Apply max_train_samples limit if specified
    if args.max_train_samples and args.max_train_samples < len(X_train):
        logger.info(f"Limiting training data to {args.max_train_samples} samples (from {len(X_train)} available) using {args.sampling_strategy} sampling")
        
        # Use deterministic sampling based on the dataset-specific seed
        np.random.seed(dataset_specific_seed)
        
        if args.sampling_strategy == "balanced":
            # Balanced sampling: equal samples per class
            unique_classes = np.unique(y_train)
            num_classes = len(unique_classes)
            
            # Calculate how many examples to take from each class
            target_samples = args.max_train_samples
            examples_per_class = target_samples // num_classes
            remainder = target_samples % num_classes
            
            # Initialize list to hold selected indices
            selected_indices = []
            
            for i, class_label in enumerate(unique_classes):
                # Get indices for this class
                class_indices = np.where(y_train == class_label)[0]
                
                # Determine how many samples to take from this class
                samples_from_class = examples_per_class + (1 if i < remainder else 0)
                samples_from_class = min(samples_from_class, len(class_indices))
                
                # Randomly sample from this class
                if samples_from_class > 0:
                    chosen_indices = np.random.choice(class_indices, samples_from_class, replace=False)
                    selected_indices.extend(chosen_indices)
                    
                logger.info(f"  Class {class_label}: selected {samples_from_class} out of {len(class_indices)} samples")
            
            # Convert to numpy array and sort for consistency
            indices = np.array(selected_indices)
            indices.sort()
        else:
            # Random sampling: simple random selection
            indices = np.random.choice(len(X_train), args.max_train_samples, replace=False)
            indices.sort()  # Sort to maintain some consistency
        
        # Apply the selected indices
        X_train = X_train.iloc[indices] if hasattr(X_train, 'iloc') else X_train[indices]
        y_train = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
        logger.info(f"Training data limited to {len(X_train)} samples")
    
    # Skip embeddings if only running baselines
    if args.baselines_only:
        logger.info(f"Skipping TabPFN embeddings for dataset {dataset['name']} (baselines_only mode)")
        
        # Add processed data to the dataset dictionary without embeddings
        dataset.update({
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "is_classification": is_classification
        })
    else:
        # Get TabPFN embeddings
        logger.info(f"Computing TabPFN embeddings for dataset {dataset['name']} with size {args.embedding_size}")
        
        # Handle embedding cache directory
        cache_dir = None
        if args.embedding_cache_dir.lower() != 'none':
            cache_dir = args.embedding_cache_dir
            # Create the directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
        # Use dataset ID or name as cache identifier
        dataset_identifier = str(dataset["id"])
        
        # Check if we have cached embeddings and whether they need subsetting based on max_test_samples
        cache_file = None
        if cache_dir and not args.force_recompute_embeddings:
            from clam.data.embeddings import generate_dataset_hash
            dataset_hash = generate_dataset_hash(X_train, y_train, args.embedding_size, dataset_identifier)
            prefix = f"{dataset_identifier}_"
            cache_file = os.path.join(cache_dir, f"{prefix}tabpfn_embeddings_{dataset_hash}.npz")
            
            if os.path.exists(cache_file):
                logger.info(f"Found cached embeddings at {cache_file}")
                try:
                    # Use the utility function to load embeddings with limit
                    train_embeddings, val_embeddings, test_embeddings, y_train_sample = load_embeddings_with_limit(
                        cache_file, args.max_test_samples
                    )
                    
                    # Create dummy TabPFN model since we loaded from cache
                    from tabpfn import TabPFNClassifier
                    tabpfn = TabPFNClassifier(
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        n_estimators=8,
                        ignore_pretraining_limits=True
                    )
                    
                    logger.info(f"Loaded cached embeddings - Train: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load cached embeddings: {e}. Falling back to regular loading.")
                    # If cache loading fails, fall back to regular computation
                    train_embeddings, val_embeddings, test_embeddings, tabpfn, y_train_sample = get_tabpfn_embeddings(
                        X_train, y_train, X_val, X_test,
                        embedding_size=args.embedding_size,
                        cache_dir=cache_dir,
                        dataset_name=dataset_identifier,
                        force_recompute=args.force_recompute_embeddings,
                        seed=dataset_specific_seed
                    )
            else:
                # No cache file exists, compute embeddings
                train_embeddings, val_embeddings, test_embeddings, tabpfn, y_train_sample = get_tabpfn_embeddings(
                    X_train, y_train, X_val, X_test,
                    embedding_size=args.embedding_size,
                    cache_dir=cache_dir,
                    dataset_name=dataset_identifier,
                    force_recompute=args.force_recompute_embeddings,
                    seed=dataset_specific_seed
                )
        else:
            # No cache dir or force recompute
            train_embeddings, val_embeddings, test_embeddings, tabpfn, y_train_sample = get_tabpfn_embeddings(
                X_train, y_train, X_val, X_test,
                embedding_size=args.embedding_size,
                cache_dir=cache_dir,
                dataset_name=dataset_identifier,
                force_recompute=args.force_recompute_embeddings,
                seed=dataset_specific_seed
            )

        # Add processed data to the dataset dictionary with embeddings
        dataset.update({
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "train_embeddings": train_embeddings,
            "val_embeddings": val_embeddings,
            "test_embeddings": test_embeddings,
            "y_train_sample": y_train_sample,
            "is_classification": is_classification
        })
    
    return dataset

def compute_label_frequency_mapping(true_labels: np.ndarray, predicted_labels: np.ndarray, threshold: float = 0.05) -> Optional[Dict[int, int]]:
    """
    Compute the optimal label mapping from predicted labels to true labels based on frequency matching.
    Uses the Hungarian algorithm to find the optimal permutation that minimizes total frequency difference.
    
    Args:
        true_labels: Ground truth labels
        predicted_labels: Model predictions  
        threshold: Minimum frequency difference threshold to trigger label remapping
        
    Returns:
        Dictionary mapping predicted label to true label, or None if no remapping needed
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Get unique labels and their frequencies
    true_unique, true_counts = np.unique(true_labels, return_counts=True)
    pred_unique, pred_counts = np.unique(predicted_labels, return_counts=True)
    
    # Compute frequencies
    true_freq = true_counts / len(true_labels)
    pred_freq = pred_counts / len(predicted_labels)
    
    # Handle case where predicted labels don't include all true labels
    n_classes = max(len(true_unique), len(pred_unique))
    
    # Create frequency arrays with zeros for missing labels
    true_freq_full = np.zeros(n_classes)
    pred_freq_full = np.zeros(n_classes)
    
    for i, label in enumerate(true_unique):
        if label < n_classes:
            true_freq_full[label] = true_freq[i]
    
    for i, label in enumerate(pred_unique):
        if label < n_classes:
            pred_freq_full[label] = pred_freq[i]
    
    # Create cost matrix (frequency differences)
    cost_matrix = np.abs(true_freq_full[:, np.newaxis] - pred_freq_full[np.newaxis, :])
    
    # Find optimal assignment using Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Create mapping
    label_mapping = dict(zip(col_indices, row_indices))
    
    # Check if remapping is needed (any frequency difference exceeds threshold)
    max_diff = np.max(cost_matrix[row_indices, col_indices])
    
    logger.info(f"Label frequency analysis:")
    logger.info(f"True label frequencies: {dict(zip(true_unique, true_freq))}")
    logger.info(f"Predicted label frequencies: {dict(zip(pred_unique, pred_freq))}")
    logger.info(f"Maximum frequency difference after optimal mapping: {max_diff:.4f}")
    
    if max_diff > threshold:
        logger.info(f"Label remapping triggered (max diff {max_diff:.4f} > threshold {threshold})")
        logger.info(f"Label mapping: {label_mapping}")
        return label_mapping
    else:
        logger.info(f"No label remapping needed (max diff {max_diff:.4f} <= threshold {threshold})")
        return None

def apply_label_mapping(predictions: np.ndarray, label_mapping: Dict[int, int]) -> np.ndarray:
    """
    Apply label mapping to predictions.
    
    Args:
        predictions: Array of predicted labels
        label_mapping: Dictionary mapping from predicted to true labels
        
    Returns:
        Remapped predictions
    """
    # Ensure predictions is a numpy array
    predictions = np.array(predictions)
    remapped = predictions.copy()
    
    for pred_label, true_label in label_mapping.items():
        mask = predictions == pred_label
        remapped[mask] = true_label
    
    return remapped

def compute_frequency_distribution(labels: np.ndarray, label_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute frequency distribution of labels.
    
    Args:
        labels: Array of labels
        label_names: Optional list of label names, if available
        
    Returns:
        Dictionary with frequency distribution information
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    frequencies = counts / len(labels)
    
    # Create label name mapping
    if label_names is not None:
        label_map = {i: name for i, name in enumerate(label_names)}
    else:
        label_map = {label: str(label) for label in unique_labels}
    
    distribution = {
        'counts': {label_map.get(label, str(label)): int(count) for label, count in zip(unique_labels, counts)},
        'frequencies': {label_map.get(label, str(label)): float(freq) for label, freq in zip(unique_labels, frequencies)},
        'total_samples': len(labels)
    }
    
    return distribution

def compute_baseline_probabilities(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: Dict[str, Any],
    class_token_ids: List[int],
    prefix_start_id: int,
    prefix_end_id: int,
    prefix_data_file: str,
    sample_ratio: float = 0.1,
    max_samples: int = 1000
) -> np.ndarray:
    """
    Compute baseline probabilities for each class token from a sample of training data.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        dataset: Dataset containing training data
        class_token_ids: List of token IDs for classes
        prefix_start_id: Token ID for prefix start
        prefix_end_id: Token ID for prefix end
        prefix_data_file: Path to prefix data file
        sample_ratio: Ratio of training data to sample
        max_samples: Maximum number of samples to use
        
    Returns:
        Array of baseline probabilities for each class
    """
    import logging
    from clam.data import create_llm_dataset
    from clam.train import evaluate_llm_on_test_set
    
    logger = logging.getLogger(__name__)
    logger.info(f"Computing baseline probabilities from training sample")
    
    # Sample from training data
    # Note: embeddings may have fewer rows than X_train, so we need to sample from embedding indices
    n_embeddings = len(dataset["train_embeddings"])
    n_samples = min(int(n_embeddings * sample_ratio), max_samples, n_embeddings)
    
    # Create a deterministic RNG for this sampling to ensure reproducibility
    rng = np.random.RandomState(seed=42)  # Use fixed seed for consistent baseline calibration
    sample_indices = rng.choice(n_embeddings, size=n_samples, replace=False)
    
    # Use the same indices for X, y, and embeddings
    X_sample = dataset["X_train"][:n_embeddings][sample_indices]
    y_sample = dataset["y_train"][:n_embeddings][sample_indices]
    embeddings_sample = dataset["train_embeddings"][sample_indices]
    
    # Create temporary dataset for baseline computation - use arrays with proper shape but zero rows
    empty_X = np.empty((0, X_sample.shape[1]))
    empty_y = np.array([])
    empty_embeddings = np.empty((0, embeddings_sample.shape[1]))
    
    baseline_dataset, _, _, label_encoder, _ = create_llm_dataset(
        X_sample, y_sample,
        empty_X, empty_y,  # Empty validation set
        empty_X, empty_y,  # Empty test set
        embeddings_sample, empty_embeddings, empty_embeddings,
        tokenizer, prefix_start_id, prefix_end_id, class_token_ids,
        output_dir="/tmp/baseline_calibration",
        num_few_shot_examples=10  # Use fewer examples for faster computation during calibration
    )
    
    # Get probabilities for each sample
    logger.info(f"Getting probabilities for {len(baseline_dataset)} samples")
    
    # Evaluate and extract raw probabilities
    results = evaluate_llm_on_test_set(
        model, tokenizer, baseline_dataset,
        label_encoder, prefix_start_id, prefix_end_id,
        class_token_ids, prefix_data_file,
        max_test_samples=len(baseline_dataset),
        return_raw_probabilities=True  # We'll need to add this parameter
    )
    
    # Extract raw probabilities matrix if available
    if 'raw_probabilities' in results:
        # raw_probabilities should be shape (n_samples, n_classes)
        prob_matrix = np.array(results['raw_probabilities'])
        baseline_probs = np.mean(prob_matrix, axis=0)
        
        logger.info(f"Baseline probabilities: {baseline_probs}")
        return baseline_probs
    else:
        logger.warning("Raw probabilities not available, using uniform baseline")
        return np.ones(len(class_token_ids)) / len(class_token_ids)

def evaluate_dataset(dataset: Dict[str, Any], model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids, args):
    """
    Evaluate a model on a single dataset.
    
    Args:
        dataset: Dictionary with processed dataset information
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        prefix_start_id: Token ID for prefix start marker
        prefix_end_id: Token ID for prefix end marker
        class_token_ids: List of token IDs for class tokens
        args: Command line arguments
        
    Returns:
        Dictionary with evaluation results
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Note: load_embeddings_with_limit has been moved to module level
    
    # Create dataset-specific output directory
    dataset_output_dir = os.path.join(args.output_dir, f"dataset_{dataset['id']}")
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Log dataset info
    logger.info(f"Evaluating on dataset: {dataset['name']} (ID: {dataset['id']})")
    
    # Save dataset information for reference
    dataset_info = {
        "id": dataset["id"],
        "name": dataset["name"],
        "num_samples": len(dataset["X_train"]),
        "num_features": dataset["X_train"].shape[1],
        "num_classes": len(np.unique(dataset["y_train"])),
        "class_distribution": {
            str(cls): int(count) 
            for cls, count in zip(*np.unique(dataset["y_train"], return_counts=True))
        },
        "is_csv": dataset.get("is_csv", False)  # Track whether this is a CSV dataset
    }
    
    # Save this information
    with open(os.path.join(dataset_output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    # Create LLM dataset
    logger.info(f"Creating LLM dataset for {dataset['name']}")
    train_dataset, eval_dataset, test_dataset, label_encoder, prefix_data_file = create_llm_dataset(
        dataset["X_train"], dataset["y_train_sample"], 
        dataset["X_val"], dataset["y_val"], 
        dataset["X_test"], dataset["y_test"],
        dataset["train_embeddings"], dataset["val_embeddings"], dataset["test_embeddings"],
        tokenizer, prefix_start_id, prefix_end_id, class_token_ids,
        output_dir=dataset_output_dir,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples
    )
    
    # Label fitting: Create holdout set if needed
    label_mapping = None
    if args.label_fitting:
        logger.info(f"Label fitting enabled with holdout ratio {args.label_fitting_holdout_ratio}")
        
        # Create holdout set from training data
        # Note: embeddings may have fewer rows than X_train, so we need to sample from embedding indices
        n_embeddings = len(dataset["train_embeddings"])
        n_holdout = min(int(n_embeddings * args.label_fitting_holdout_ratio), n_embeddings)
        
        # Use a deterministic RNG based on dataset ID and global seed for consistent holdout sets
        # Use hashlib for stable hashing across Python sessions
        dataset_id_bytes = str(dataset["id"]).encode('utf-8')
        dataset_id_hash = int(hashlib.md5(dataset_id_bytes).hexdigest()[:8], 16) % 10000
        holdout_seed = args.seed + dataset_id_hash
        holdout_rng = np.random.RandomState(seed=holdout_seed)
        holdout_indices = holdout_rng.choice(n_embeddings, size=n_holdout, replace=False)
        
        # Use the same indices for X, y, and embeddings
        X_holdout = dataset["X_train"][:n_embeddings][holdout_indices]
        y_holdout = dataset["y_train"][:n_embeddings][holdout_indices]
        embeddings_holdout = dataset["train_embeddings"][holdout_indices]
        
        # Create holdout dataset - use arrays with proper shape but zero rows
        empty_X = np.empty((0, X_holdout.shape[1]))
        empty_y = np.array([])
        empty_embeddings = np.empty((0, embeddings_holdout.shape[1]))
        
        # Create holdout dataset
        holdout_dataset, _, _, _, _ = create_llm_dataset(
            X_holdout, y_holdout,
            empty_X, empty_y,  # Empty validation set
            empty_X, empty_y,  # Empty test set
            embeddings_holdout, empty_embeddings, empty_embeddings,
            tokenizer, prefix_start_id, prefix_end_id, class_token_ids,
            output_dir=dataset_output_dir,
            max_train_samples=args.max_train_samples
        )
        
        # Get predictions on holdout set
        logger.info(f"Getting predictions on {len(holdout_dataset)} holdout samples for label fitting")
        holdout_results = evaluate_llm_on_test_set(
            model, tokenizer, holdout_dataset,
            label_encoder, prefix_start_id, prefix_end_id,
            class_token_ids, prefix_data_file,
            max_test_samples=len(holdout_dataset)
        )
        
        # Compute label mapping
        if 'predictions' in holdout_results:
            label_mapping = compute_label_frequency_mapping(
                y_holdout, 
                holdout_results['predictions'],
                threshold=args.label_fitting_threshold
            )
    
    # Compute baseline probabilities if requested
    baseline_probs = None
    if args.baseline_calibration:
        logger.info("Computing baseline probabilities for calibration")
        baseline_probs = compute_baseline_probabilities(
            model, tokenizer, dataset,
            class_token_ids, prefix_start_id, prefix_end_id,
            prefix_data_file,
            sample_ratio=args.baseline_sample_ratio
        )
    
    # Compute class frequencies for minority boosting
    class_weights = None
    if args.minority_class_boost:
        logger.info("Computing class frequencies for minority boosting")
        # Get class frequencies from training data
        unique_classes, class_counts = np.unique(dataset["y_train"], return_counts=True)
        class_frequencies = class_counts / len(dataset["y_train"])
        
        # Compute inverse frequency weights (higher weight for rarer classes)
        inverse_frequencies = 1.0 / (class_frequencies + 1e-6)
        
        # Normalize weights
        inverse_frequencies = inverse_frequencies / np.mean(inverse_frequencies)
        
        # Apply boost factor
        class_weights = np.power(inverse_frequencies, args.minority_boost_factor)
        
        # Create weight array for all possible classes
        max_classes = len(class_token_ids)
        full_weights = np.ones(max_classes)
        for i, cls in enumerate(unique_classes):
            if cls < max_classes:
                full_weights[cls] = class_weights[i]
        
        class_weights = full_weights
        logger.info(f"Class weights: {class_weights}")
    
    # Evaluate on test set
    logger.info(f"Evaluating on {len(test_dataset)} test samples")
    
    # By default, only consider classes present in ground truth
    # The evaluator will auto-detect these when allowed_classes=None
    results = evaluate_llm_on_test_set(
        model, tokenizer, test_dataset,
        label_encoder, prefix_start_id, prefix_end_id,
        class_token_ids, prefix_data_file, 
        max_test_samples=args.max_test_samples,
        baseline_probabilities=baseline_probs,
        score_normalization=args.score_normalization,
        normalization_temperature=args.normalization_temperature,
        class_weights=class_weights,
        allowed_classes=None  # Auto-detect from test dataset
    )
    
    # Apply label mapping if needed
    if label_mapping is not None and 'predictions' in results:
        original_accuracy = results['accuracy']
        remapped_predictions = apply_label_mapping(results['predictions'], label_mapping)
        
        # Recompute metrics with remapped predictions
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
        y_test = dataset["y_test"][:len(results['predictions'])]  # Ensure same length
        
        remapped_accuracy = accuracy_score(y_test, remapped_predictions)
        results['accuracy'] = remapped_accuracy
        results['predictions'] = remapped_predictions
        results['label_mapping'] = label_mapping
        results['original_accuracy'] = original_accuracy
        
        # Update classification report and confusion matrix
        results['classification_report'] = classification_report(y_test, remapped_predictions, output_dict=True)
        results['confusion_matrix'] = confusion_matrix(y_test, remapped_predictions)
        
        logger.info(f"Applied label mapping: accuracy changed from {original_accuracy:.4f} to {remapped_accuracy:.4f}")
    
    # Get ground truth labels
    y_test = dataset["y_test"][:len(test_dataset)]
    
    # Store predictions and ground truth for distribution calculation
    predictions = results.get('predictions', [])
    
    # Compute frequency distributions
    prediction_distribution = None
    ground_truth_distribution = None
    
    if len(predictions) > 0:
        # Try to get label names if available from the label encoder
        label_names = None
        if hasattr(label_encoder, 'classes_'):
            label_names = [str(cls) for cls in label_encoder.classes_]
        
        # Compute distributions
        prediction_distribution = compute_frequency_distribution(predictions, label_names)
        ground_truth_distribution = compute_frequency_distribution(y_test, label_names)
        
        # Log the distributions
        logger.info(f"Prediction frequency distribution: {prediction_distribution['frequencies']}")
        logger.info(f"Ground truth frequency distribution: {ground_truth_distribution['frequencies']}")
    
    # Log results
    logger.info(f"Test accuracy on {dataset['name']}: {results['accuracy']:.4f}")
    if 'balanced_accuracy' in results and results['balanced_accuracy'] is not None:
        logger.info(f"Test balanced accuracy on {dataset['name']}: {results['balanced_accuracy']:.4f}")
    if 'roc_auc' in results and results['roc_auc'] is not None:
        logger.info(f"Test ROC AUC on {dataset['name']}: {results['roc_auc']:.4f}")
    
    # Save results
    result_summary = {
        'dataset': dataset['name'],
        'dataset_id': dataset['id'],
        'model_path': args.model_path,
        'accuracy': float(results['accuracy']),
        'num_samples': len(test_dataset),
    }
    
    # Add balanced accuracy if available
    if 'balanced_accuracy' in results and results['balanced_accuracy'] is not None:
        result_summary['balanced_accuracy'] = float(results['balanced_accuracy'])
        
    # Add ROC AUC if available
    if 'roc_auc' in results and results['roc_auc'] is not None:
        result_summary['roc_auc'] = float(results['roc_auc'])
    
    # Add frequency distributions
    if prediction_distribution is not None:
        result_summary['prediction_distribution'] = prediction_distribution
        result_summary['ground_truth_distribution'] = ground_truth_distribution
    
    # Add label fitting information if applied
    if args.label_fitting:
        result_summary['label_fitting'] = {
            'enabled': True,
            'threshold': args.label_fitting_threshold,
            'holdout_ratio': args.label_fitting_holdout_ratio
        }
        if label_mapping is not None:
            result_summary['label_fitting']['applied'] = True
            result_summary['label_fitting']['mapping'] = label_mapping
            result_summary['label_fitting']['original_accuracy'] = float(results.get('original_accuracy', results['accuracy']))
            result_summary['label_fitting']['remapped_accuracy'] = float(results['accuracy'])
        else:
            result_summary['label_fitting']['applied'] = False
    
    # Add baseline calibration information if applied
    if args.baseline_calibration:
        result_summary['baseline_calibration'] = {
            'enabled': True,
            'sample_ratio': args.baseline_sample_ratio,
            'baseline_probabilities': baseline_probs.tolist() if baseline_probs is not None else None
        }
    
    # Add classification report if available
    if 'classification_report' in results:
        result_summary['classification_report'] = results['classification_report']
    
    # Add confusion matrix if available
    if 'confusion_matrix' in results:
        result_summary['confusion_matrix'] = results['confusion_matrix'].tolist() if not isinstance(results['confusion_matrix'], list) else results['confusion_matrix']
    
    # Save to JSON
    results_file = os.path.join(dataset_output_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(result_summary, f, indent=2)
    
    # Return results
    return_dict = {
        'dataset_id': dataset['id'],
        'dataset_name': dataset['name'],
        'accuracy': results['accuracy'],
        'balanced_accuracy': results.get('balanced_accuracy'),
        'classification_report': results.get('classification_report'),
        'confusion_matrix': results.get('confusion_matrix'),
        'roc_auc': results.get('roc_auc'),
        'results_file': results_file
    }
    
    # Add frequency distributions if available
    if prediction_distribution is not None:
        return_dict['prediction_distribution'] = prediction_distribution
        return_dict['ground_truth_distribution'] = ground_truth_distribution
    
    # Store predictions for aggregated analysis
    if 'predictions' in results:
        return_dict['predictions'] = results['predictions'].tolist() if hasattr(results['predictions'], 'tolist') else results['predictions']
        return_dict['ground_truth'] = y_test.tolist() if hasattr(y_test, 'tolist') else y_test
    
    # Add label fitting information if used
    if args.label_fitting and 'label_fitting' in result_summary:
        return_dict['label_fitting'] = result_summary['label_fitting']
    
    # Add baseline calibration information if used
    if args.baseline_calibration and 'baseline_calibration' in result_summary:
        return_dict['baseline_calibration'] = result_summary['baseline_calibration']
    
    return return_dict

def create_and_evaluate_baseline_model(model_name, dataset, args):
    """
    Create and evaluate a baseline model on a dataset.
    
    Args:
        model_name: Name of the model ('catboost', 'tabpfn_v2', 'random_forest', etc.)
        dataset: Dictionary with processed dataset information
        args: Command line arguments
        
    Returns:
        Dictionary with evaluation results
    """
    import logging
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score
    import numpy as np
    import time
    
    logger = logging.getLogger(__name__)
    logger.info(f"Creating and evaluating {model_name} model on dataset {dataset['name']}")
    
    # Extract dataset components
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    
    # For CatBoost, we need to preprocess the data preserving categorical features
    if model_name == 'catboost':
        # Get the categorical indicators
        categorical_indicator = dataset.get('categorical_indicator_raw', [False] * X_train.shape[1])
        
        # For CatBoost, we need to reconstruct the data preserving categorical features
        # We'll apply the preserve_categorical preprocessing to the already-split data
        
        # Apply preprocessing that preserves categorical features
        X_train_processed = preprocess_features(X_train, categorical_indicator, preserve_categorical=True)
        X_test_processed = preprocess_features(X_test, categorical_indicator, preserve_categorical=True)
        
        # Update the variables to use the categorical-preserving data
        X_train = X_train_processed
        X_test = X_test_processed
        
        logger.info(f"Using CatBoost-specific preprocessing: Train shape {X_train.shape}, Test shape {X_test.shape}")
    
    # Training data was already limited earlier at the dataset level if max_train_samples was specified
    # No additional limiting needed here - all baseline models use the same pre-limited training data
    logger.info(f"Using training data with {len(X_train)} samples for baseline model training")
    
    # Limit test samples if specified
    if args.max_test_samples and args.max_test_samples < len(X_test):
        X_test = X_test[:args.max_test_samples]
        y_test = y_test[:args.max_test_samples]
        logger.info(f"Limited test set to {args.max_test_samples} samples")
    
    start_time = time.time()
    
    # Initialize and train model based on model_name
    if model_name == 'catboost':
        try:
            from catboost import CatBoostClassifier
            
            # Get categorical features for CatBoost
            categorical_features = []
            if 'categorical_indicator_raw' in dataset:
                # Find indices of categorical features
                max_features = X_train.shape[1]
                categorical_features = [i for i, is_cat in enumerate(dataset['categorical_indicator_raw']) 
                                      if i < max_features and is_cat]
                logger.info(f"CatBoost using {len(categorical_features)} categorical features: {categorical_features}")
                
                # Log data types for debugging
                import pandas as pd
                df_train = pd.DataFrame(X_train)
                for cat_idx in categorical_features[:5]:  # Log first 5 categorical features
                    if cat_idx < df_train.shape[1]:
                        logger.info(f"  Feature {cat_idx}: dtype={df_train.iloc[:, cat_idx].dtype}, "
                                  f"unique values={df_train.iloc[:, cat_idx].nunique()}, "
                                  f"sample values={df_train.iloc[:, cat_idx].unique()[:3].tolist()}")
            
            model = CatBoostClassifier(
                iterations=args.catboost_iterations,
                depth=args.catboost_depth,
                learning_rate=args.catboost_learning_rate,
                random_seed=args.seed,
                verbose=False
            )
            
            # Train the model
            model.fit(
                X_train, y_train, 
                cat_features=categorical_features,
                verbose=False
            )
            
        except ImportError:
            logger.error("CatBoost not installed. Please install it with 'pip install catboost'.")
            return {
                'model_name': model_name,
                'dataset_name': dataset['name'],
                'error': "CatBoost not installed. Please install it with 'pip install catboost'."
            }
    
    elif model_name == 'tabpfn_v2':
        try:
            # Check if we can import from the ticl package
            from tabpfn import TabPFNClassifier
            
            # Initialize the model
            model = TabPFNClassifier(
                device=args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu',
                n_estimators=args.tabpfn_v2_N_ensemble_configurations,
                ignore_pretraining_limits=True,
            )
            
            # Train (fit) the model
            model.fit(X_train, y_train)
            
        except ImportError as e:
            logger.error(f"TabPFN v2 not properly installed: {e}")
            return {
                'model_name': model_name,
                'dataset_name': dataset['name'],
                'error': f"TabPFN v2 not properly installed: {e}"
            }
    
    elif model_name == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            random_state=args.seed,
            n_jobs=-1  # Use all available cores
        )
        
        # Train the model
        model.fit(X_train, y_train)
    
    elif model_name == 'gradient_boosting':
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Limit features to maximum of 500
        max_features = 500
        if X_train.shape[1] > max_features:
            logger.info(f"Limiting gradient boosting features from {X_train.shape[1]} to {max_features}")
            feature_selector = SelectKBest(score_func=f_classif, k=max_features)
            X_train_selected = feature_selector.fit_transform(X_train, y_train)
            X_test = feature_selector.transform(X_test)
            logger.info(f"Selected {X_train_selected.shape[1]} features for gradient boosting")
        else:
            X_train_selected = X_train
            logger.info(f"Using all {X_train.shape[1]} features for gradient boosting (under limit)")
        
        model = GradientBoostingClassifier(
            n_estimators=args.gb_n_estimators,
            learning_rate=args.gb_learning_rate,
            random_state=args.seed
        )
        
        # Train the model
        model.fit(X_train_selected, y_train)
    
    elif model_name == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = LogisticRegression(
            max_iter=args.lr_max_iter,
            C=args.lr_C,
            random_state=args.seed,
            n_jobs=-1  # Use all available cores
        )
        
        # Train the model
        model.fit(X_train_scaled, y_train)
    
    else:
        logger.error(f"Unknown model name: {model_name}")
        return {
            'model_name': model_name,
            'dataset_name': dataset['name'],
            'error': f"Unknown model name: {model_name}"
        }
    
    # Measure training time
    training_time = time.time() - start_time
    logger.info(f"Training time for {model_name}: {training_time:.2f} seconds")
    
    # Make predictions
    start_time = time.time()
    
    if model_name == 'logistic_regression':
        # Use the scaled test data
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Measure prediction time
    prediction_time = time.time() - start_time
    logger.info(f"Prediction time for {model_name}: {prediction_time:.2f} seconds")
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate balanced accuracy
    try:
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        logger.info(f"Accuracy for {model_name} on {dataset['name']}: {accuracy:.4f}, Balanced accuracy: {balanced_acc:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute balanced accuracy: {e}")
        balanced_acc = None
        logger.info(f"Accuracy for {model_name} on {dataset['name']}: {accuracy:.4f}")
    
    # Calculate ROC AUC if probabilities are available
    roc_auc = None
    if y_prob is not None:
        try:
            # Get unique classes
            unique_classes = np.unique(y_test)
            
            # For binary classification
            if len(unique_classes) == 2:
                # Get probabilities for the positive class (usually class 1)
                pos_class_idx = 1 if 1 in unique_classes else unique_classes[1]
                binary_truth = np.array([1 if y == pos_class_idx else 0 for y in y_test])
                
                # Check if class index exists in probability array
                if y_prob.shape[1] > pos_class_idx:
                    binary_probs = y_prob[:, pos_class_idx]
                    roc_auc = roc_auc_score(binary_truth, binary_probs)
                    logger.info(f"ROC AUC for {model_name} on {dataset['name']}: {roc_auc:.4f}")
            # For multi-class classification
            elif len(unique_classes) > 2:
                # Use one-vs-rest approach
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_test, classes=unique_classes)
                
                # Make sure we have probabilities for all classes
                if y_prob.shape[1] >= len(unique_classes):
                    # Get probabilities for the classes that are present
                    probs_array = np.array([y_prob[:, i] for i in unique_classes]).T
                    roc_auc = roc_auc_score(y_bin, probs_array, multi_class='ovr')
                    logger.info(f"ROC AUC (OVR) for {model_name} on {dataset['name']}: {roc_auc:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute ROC AUC: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    
    # Generate classification report and confusion matrix
    try:
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Generated classification report and confusion matrix for {model_name}")
    except Exception as e:
        logger.warning(f"Could not generate detailed metrics: {e}")
        report = None
        cm = None
    
    # Compute frequency distributions
    prediction_distribution = compute_frequency_distribution(y_pred)
    ground_truth_distribution = compute_frequency_distribution(y_test)
    
    # Log the distributions
    logger.info(f"Prediction frequency distribution: {prediction_distribution['frequencies']}")
    logger.info(f"Ground truth frequency distribution: {ground_truth_distribution['frequencies']}")
    
    # Calculate total time
    total_time = training_time + prediction_time
    
    # Return results
    results = {
        'model_name': model_name,
        'dataset_name': dataset['name'],
        'dataset_id': dataset['id'],
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc) if balanced_acc is not None else None,
        'training_time': float(training_time),
        'prediction_time': float(prediction_time),
        'total_time': float(total_time),
        'num_train_samples': len(X_train),
        'num_test_samples': len(X_test),
        'num_features': X_train.shape[1],
        'num_classes': len(np.unique(y_train)),
        'prediction_distribution': prediction_distribution,
        'ground_truth_distribution': ground_truth_distribution,
        'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
        'ground_truth': y_test.tolist() if hasattr(y_test, 'tolist') else y_test
    }
    
    # Add ROC AUC if available
    if roc_auc is not None:
        results['roc_auc'] = float(roc_auc)
    
    if report is not None:
        results['classification_report'] = report
    
    if cm is not None:
        results['confusion_matrix'] = cm.tolist()
    
    return results

def create_and_evaluate_llm_model(model_id, dataset, args, cached_model=None):
    """
    Create and evaluate an LLM model from model_id on a dataset.
    
    Args:
        model_id: Hugging Face model ID
        dataset: Dictionary with processed dataset information
        args: Command line arguments
        cached_model: Optional tuple of (model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids)
        
    Returns:
        Dictionary with evaluation results
    """
    import logging
    import time
    from clam.models import prepare_qwen_with_prefix_embedding
    
    logger = logging.getLogger(__name__)
    logger.info(f"Creating and evaluating LLM model {model_id} on dataset {dataset['name']}")
    
    start_time = time.time()
    
    # 1. Prepare the LLM model
    if cached_model is not None:
        # Use the cached model
        model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = cached_model
        logger.info(f"Using cached model {model_id}")
    else:
        # Load the model (backward compatibility)
        try:
            model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = prepare_qwen_with_prefix_embedding(
                embedding_size=args.embedding_size,
                model_id=model_id
            )
            
            logger.info(f"Successfully loaded model {model_id}")
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return {
                'model_name': model_id,
                'dataset_name': dataset['name'],
                'error': f"Error loading model: {e}"
            }
    
    # 2. Create dataset for evaluation
    dataset_output_dir = os.path.join(args.output_dir, f"dataset_{dataset['id']}")
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    train_dataset, eval_dataset, test_dataset, label_encoder, prefix_data_file = create_llm_dataset(
        dataset["X_train"], dataset["y_train_sample"], 
        dataset["X_val"], dataset["y_val"], 
        dataset["X_test"], dataset["y_test"],
        dataset["train_embeddings"], dataset["val_embeddings"], dataset["test_embeddings"],
        tokenizer, prefix_start_id, prefix_end_id, class_token_ids,
        output_dir=dataset_output_dir,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples
    )
    
    # 3. Evaluate on test set
    logger.info(f"Evaluating on {len(test_dataset)} test samples")
    results = evaluate_llm_on_test_set(
        model, tokenizer, test_dataset,
        label_encoder, prefix_start_id, prefix_end_id,
        class_token_ids, prefix_data_file, 
        max_test_samples=args.max_test_samples,
        allowed_classes=None  # Auto-detect from test dataset
    )
    
    # Total time
    total_time = time.time() - start_time
    
    # Get predictions and ground truth
    y_test = dataset["y_test"][:len(test_dataset)]
    predictions = results.get('predictions', [])
    
    # Compute frequency distributions
    if len(predictions) > 0:
        prediction_distribution = compute_frequency_distribution(predictions)
        ground_truth_distribution = compute_frequency_distribution(y_test)
        
        # Log the distributions
        logger.info(f"Prediction frequency distribution: {prediction_distribution['frequencies']}")
        logger.info(f"Ground truth frequency distribution: {ground_truth_distribution['frequencies']}")
    else:
        prediction_distribution = None
        ground_truth_distribution = None
    
    # Format results
    formatted_results = {
        'model_name': model_id,
        'dataset_name': dataset['name'],
        'dataset_id': dataset['id'],
        'accuracy': float(results['accuracy']),
        'total_time': float(total_time),
        'num_train_samples': len(dataset['X_train']),
        'num_test_samples': len(dataset['X_test']),
        'num_features': dataset['X_train'].shape[1],
        'num_classes': len(np.unique(dataset['y_train']))
    }
    
    # Add frequency distributions
    if prediction_distribution is not None:
        formatted_results['prediction_distribution'] = prediction_distribution
        formatted_results['ground_truth_distribution'] = ground_truth_distribution
        formatted_results['predictions'] = predictions
        formatted_results['ground_truth'] = y_test.tolist() if hasattr(y_test, 'tolist') else y_test
    
    # Add classification report if available
    if 'classification_report' in results:
        formatted_results['classification_report'] = results['classification_report']
    
    # Add confusion matrix if available
    if 'confusion_matrix' in results:
        formatted_results['confusion_matrix'] = results['confusion_matrix'].tolist() if not isinstance(results['confusion_matrix'], list) else results['confusion_matrix']
    
    logger.info(f"Evaluation complete for {model_id}. Accuracy: {formatted_results['accuracy']:.4f}")
    
    return formatted_results

def main():
    args = parse_args()
    
    # Validate arguments
    if not args.baselines_only and not args.model_path and not args.model_id:
        parser = argparse.ArgumentParser()
        parser.error("Either --model_path or --model_id is required when not using --baselines_only")
    
    # Set random seed for reproducibility across all sources
    from clam.utils import set_seed
    set_seed(args.seed, deterministic=True)
    torch.backends.cudnn.benchmark = False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"evaluation_{timestamp}.log"
    logger = setup_logging(log_file=os.path.join(args.output_dir, log_filename))
    logger.info(f"Arguments: {args}")
    
    # Initialize Weights & Biases with GPU monitoring if requested
    gpu_monitor = None
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("Weights & Biases requested but not installed. Run 'pip install wandb' to install.")
        else:
            # Set up wandb run name if not provided
            if args.wandb_name is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                args.wandb_name = f"eval_{timestamp}"
            
            # Initialize wandb with GPU monitoring
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
    
    # Determine which models to evaluate
    models_to_evaluate = []
    baseline_models = ['catboost', 'tabpfn_v2', 'random_forest', 'gradient_boosting', 'logistic_regression']
    
    if args.baselines_only:
        # Only run baseline models
        if args.model_id and args.model_id.lower() in baseline_models:
            # If a specific baseline is requested
            models_to_evaluate.append(('baseline', args.model_id.lower()))
            logger.info(f"Will evaluate baseline model: {args.model_id}")
        else:
            # Run all baseline models
            for model_name in baseline_models:
                models_to_evaluate.append(('baseline', model_name))
                logger.info(f"Will evaluate baseline model: {model_name}")
    else:
        # Normal behavior when not baselines_only
        if args.model_path:
            # Evaluate the model from model_path
            models_to_evaluate.append(('clam', args.model_path))
            logger.info(f"Will evaluate CLAM model from path: {args.model_path}")
        elif args.model_id:
            # Check if it's a baseline model
            if args.model_id.lower() in baseline_models:
                models_to_evaluate.append(('baseline', args.model_id.lower()))
                logger.info(f"Will evaluate baseline model: {args.model_id}")
            else:
                # Assume it's a Hugging Face model ID
                models_to_evaluate.append(('llm', args.model_id))
                logger.info(f"Will evaluate LLM model with ID: {args.model_id}")
        
        # Add all baseline models if requested
        if args.run_all_baselines:
            for model_name in baseline_models:
                if not any(m[1] == model_name for m in models_to_evaluate):
                    models_to_evaluate.append(('baseline', model_name))
                    logger.info(f"Will also evaluate baseline model: {model_name}")
    
    # 1. Load datasets
    logger.info("Loading datasets for evaluation")
    datasets = load_datasets(args)
    logger.info(f"Loaded {len(datasets)} datasets for evaluation")
    
    # 2. Process datasets (one-time operation)
    processed_datasets = []
    for dataset in datasets:
        try:
            processed_dataset = process_dataset(dataset, args)
            processed_datasets.append(processed_dataset)
            logger.info(f"Successfully processed dataset {dataset['name']}")
        except Exception as e:
            logger.error(f"Error processing dataset {dataset['name']}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 3. Evaluate each model on each dataset
    all_results = []
    
    # Cache for loaded models to avoid reloading
    model_cache = {}
    
    for model_type, model_identifier in models_to_evaluate:
        model_results = []
        
        logger.info(f"Evaluating model {model_identifier} (type: {model_type}) on {len(processed_datasets)} datasets")
        
        # Load CLAM/LLM models once per model_identifier
        if model_type == 'clam':
            cache_key = f"clam_{model_identifier}"
            if cache_key not in model_cache:
                logger.info(f"Loading pretrained CLAM model from {model_identifier}")
                model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids, is_vq = load_pretrained_model(
                    model_identifier, 
                    device_map=args.device,
                    embedding_size=args.embedding_size,
                    model_id=args.model_id
                )
                model_cache[cache_key] = (model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids, is_vq)
            else:
                logger.info(f"Using cached CLAM model for {model_identifier}")
                model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids, is_vq = model_cache[cache_key]
        
        elif model_type == 'llm':
            cache_key = f"llm_{model_identifier}"
            if cache_key not in model_cache:
                logger.info(f"Loading LLM model {model_identifier}")
                try:
                    from clam.models import prepare_qwen_with_prefix_embedding
                    model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = prepare_qwen_with_prefix_embedding(
                        embedding_size=args.embedding_size,
                        model_id=model_identifier
                    )
                    model_cache[cache_key] = (model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids)
                    logger.info(f"Successfully loaded and cached model {model_identifier}")
                except Exception as e:
                    logger.error(f"Error loading model {model_identifier}: {e}")
                    model_cache[cache_key] = None  # Cache the failure to avoid retrying
            else:
                logger.info(f"Using cached LLM model for {model_identifier}")
                cached_value = model_cache[cache_key]
                if cached_value is None:
                    logger.error(f"Previously failed to load model {model_identifier}, skipping")
                    continue
                model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = cached_value
        
        for dataset in processed_datasets:
            try:
                if model_type == 'clam':
                    # Evaluate on dataset using the cached model
                    results = evaluate_dataset(
                        dataset, 
                        model, tokenizer, 
                        prefix_start_id, prefix_end_id, class_token_ids,
                        args
                    )
                    
                    # Add model type to results
                    results['model_type'] = 'clam_vq' if is_vq else 'clam'
                    results['model_id'] = model_identifier
                    results['is_vq'] = is_vq
                    
                elif model_type == 'baseline':
                    # Create and evaluate baseline model
                    results = create_and_evaluate_baseline_model(model_identifier, dataset, args)
                    
                    # Add model type to results
                    results['model_type'] = 'baseline'
                    
                elif model_type == 'llm':
                    # Check if model was loaded successfully
                    if model_cache.get(cache_key) is None:
                        results = {
                            'model_name': model_identifier,
                            'dataset_name': dataset['name'],
                            'error': f"Failed to load model {model_identifier}"
                        }
                    else:
                        # Create and evaluate LLM model using cached model
                        results = create_and_evaluate_llm_model(
                            model_identifier, dataset, args,
                            cached_model=(model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids)
                        )
                    
                    # Add model type to results
                    results['model_type'] = 'llm'
                
                # Add to results
                model_results.append(results)
                
                # Log to W&B if enabled using unified metrics
                if args.use_wandb and WANDB_AVAILABLE:
                    # Get model name for logging
                    model_name = model_identifier if model_type != 'baseline' else model_identifier
                    
                    # Initialize unified metrics logger
                    metrics_logger = MetricsLogger(
                        model_name=model_name,
                        dataset_name=dataset['name'],
                        use_wandb=True,
                        logger=logger
                    )
                    
                    # Prepare metrics dictionary
                    metrics_dict = results.copy()
                    
                    # Add dataset info
                    metrics_dict.update({
                        'num_features': dataset['X'].shape[1],
                        'num_samples': len(dataset['X']),
                        'num_classes': len(np.unique(dataset['y']))
                    })
                    
                    # Log all metrics using unified system
                    metrics_logger.log_all_metrics(metrics_dict)
                    
                    # Log frequency distributions
                    if 'prediction_distribution' in results and 'ground_truth_distribution' in results:
                        pred_dist = results['prediction_distribution']
                        gt_dist = results['ground_truth_distribution']
                        
                        # Log frequency distributions to console
                        logger.info(f"\\n{'='*20} FREQUENCY DISTRIBUTION SUMMARY {'='*20}")
                        logger.info(f"Dataset: {dataset['name']}, Model: {model_name}")
                        logger.info(f"Prediction Distribution: {pred_dist['frequencies']}")
                        logger.info(f"Ground Truth Distribution: {gt_dist['frequencies']}")
                        logger.info(f"{'='*66}")
                        
                        # Log to W&B - both counts and frequencies using unified metric keys
                        freq_log_dict = {}
                        for label, freq in pred_dist['frequencies'].items():
                            key = metrics_logger._get_metric_key(f"pred_freq/{label}")
                            freq_log_dict[key] = freq
                        for label, freq in gt_dist['frequencies'].items():
                            key = metrics_logger._get_metric_key(f"gt_freq/{label}")
                            freq_log_dict[key] = freq
                        for label, count in pred_dist['counts'].items():
                            key = metrics_logger._get_metric_key(f"pred_count/{label}")
                            freq_log_dict[key] = count
                        for label, count in gt_dist['counts'].items():
                            key = metrics_logger._get_metric_key(f"gt_count/{label}")
                            freq_log_dict[key] = count
                        
                        wandb.log(freq_log_dict)
                    
                    # Confusion matrix is already handled by the unified metrics system
                    # The log_all_metrics call above will handle it automatically
            
            except Exception as e:
                logger.error(f"Error evaluating {model_identifier} on dataset {dataset['name']}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Add error result
                model_results.append({
                    'model_type': model_type,
                    'model_name': model_identifier,
                    'dataset_name': dataset['name'],
                    'dataset_id': dataset['id'],
                    'error': str(e)
                })
        
        # Calculate average accuracy for this model
        valid_results = [r for r in model_results if 'accuracy' in r]
        if valid_results:
            avg_accuracy = sum(r['accuracy'] for r in valid_results) / len(valid_results)
            logger.info(f"Average accuracy for {model_identifier}: {avg_accuracy:.4f} over {len(valid_results)} datasets")
            
            # Add to all results
            all_results.extend(model_results)
            
            # Log aggregated metrics to W&B if enabled using unified metrics
            if args.use_wandb and WANDB_AVAILABLE:
                model_name = model_identifier if model_type != 'baseline' else model_identifier
                
                # Create a unified metrics logger for aggregated metrics
                # Use a special dataset name to indicate this is aggregated
                agg_metrics_logger = MetricsLogger(
                    model_name=model_name,
                    dataset_name="aggregated",
                    use_wandb=True,
                    logger=logger
                )
                
                # Calculate aggregated metrics
                aggregated_metrics = {
                    'accuracy': avg_accuracy,
                    'num_valid_datasets': len(valid_results)
                }
                
                # Calculate average balanced accuracy if available
                balanced_accs = [r.get('balanced_accuracy') for r in valid_results if r.get('balanced_accuracy') is not None]
                if balanced_accs:
                    avg_balanced_acc = sum(balanced_accs) / len(balanced_accs)
                    aggregated_metrics['balanced_accuracy'] = avg_balanced_acc
                    logger.info(f"Average balanced accuracy for {model_identifier}: {avg_balanced_acc:.4f}")
                
                # Calculate average ROC AUC if available
                roc_aucs = [r.get('roc_auc') for r in valid_results if r.get('roc_auc') is not None]
                if roc_aucs:
                    avg_roc_auc = sum(roc_aucs) / len(roc_aucs)
                    aggregated_metrics['roc_auc'] = avg_roc_auc
                    logger.info(f"Average ROC AUC for {model_identifier}: {avg_roc_auc:.4f}")
                
                # Log aggregated metrics
                agg_metrics_logger.log_all_metrics(aggregated_metrics)
        else:
            logger.warning(f"No valid results for {model_identifier}")
    
    # 4. Save all evaluation results
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy_values(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_values(v) for v in obj]
        else:
            return obj
    
    serializable_results = convert_numpy_values(all_results)
    
    all_results_file = os.path.join(args.output_dir, f"all_evaluation_results_{timestamp}.json")
    with open(all_results_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Saved all evaluation results to {all_results_file}")
    
    # 5. Create and save summary
    summary = {
        'timestamp': timestamp,
        'num_datasets': len(processed_datasets),
        'num_models': len(models_to_evaluate),
        'models': [{'type': mt, 'identifier': mi} for mt, mi in models_to_evaluate],
        'datasets': [{'id': d['id'], 'name': d['name']} for d in datasets],
        'model_summaries': []
    }
    
    # Group results by model
    for model_type, model_identifier in models_to_evaluate:
        model_results = [r for r in all_results if 
                        (r.get('model_type') == model_type and 
                         (r.get('model_id') == model_identifier or r.get('model_name') == model_identifier))]
        
        valid_results = [r for r in model_results if 'accuracy' in r]
        if valid_results:
            avg_accuracy = sum(r['accuracy'] for r in valid_results) / len(valid_results)
            
            model_summary = {
                'model_type': model_type,
                'model_identifier': model_identifier,
                'average_accuracy': float(avg_accuracy),
                'num_valid_datasets': len(valid_results),
                'dataset_accuracies': {r['dataset_name']: float(r['accuracy']) for r in valid_results}
            }
            
            # Add VQ information if available
            if model_type == 'clam' and valid_results and 'is_vq' in valid_results[0]:
                model_summary['is_vq'] = valid_results[0]['is_vq']
            
            # Add balanced accuracy information if available
            balanced_accs = [r for r in valid_results if 'balanced_accuracy' in r and r['balanced_accuracy'] is not None]
            if balanced_accs:
                avg_balanced_acc = sum(r['balanced_accuracy'] for r in balanced_accs) / len(balanced_accs)
                model_summary['average_balanced_accuracy'] = float(avg_balanced_acc)
                model_summary['dataset_balanced_accuracies'] = {r['dataset_name']: float(r['balanced_accuracy']) for r in balanced_accs}
                logger.info(f"Average balanced accuracy for {model_identifier}: {avg_balanced_acc:.4f} over {len(balanced_accs)} datasets")
            
            # Add ROC AUC information if available
            roc_aucs = [r for r in valid_results if 'roc_auc' in r and r['roc_auc'] is not None]
            if roc_aucs:
                avg_roc_auc = sum(r['roc_auc'] for r in roc_aucs) / len(roc_aucs)
                model_summary['average_roc_auc'] = float(avg_roc_auc)
                model_summary['dataset_roc_aucs'] = {r['dataset_name']: float(r['roc_auc']) for r in roc_aucs}
                logger.info(f"Average ROC AUC for {model_identifier}: {avg_roc_auc:.4f} over {len(roc_aucs)} datasets")
            
            # Add frequency distribution information
            distribution_summaries = []
            for r in valid_results:
                if 'prediction_distribution' in r and 'ground_truth_distribution' in r:
                    dist_summary = {
                        'dataset_name': r['dataset_name'],
                        'dataset_id': r['dataset_id'],
                        'prediction_distribution': r['prediction_distribution'],
                        'ground_truth_distribution': r['ground_truth_distribution']
                    }
                    distribution_summaries.append(dist_summary)
            
            if distribution_summaries:
                model_summary['frequency_distributions'] = distribution_summaries
            
            summary['model_summaries'].append(model_summary)
    
    # Save summary
    summary_file = os.path.join(args.output_dir, f"evaluation_summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved evaluation summary to {summary_file}")
    
    # Create frequency distribution summary
    logger.info("\n=== Frequency Distribution Summary ===")
    for model_type, model_identifier in models_to_evaluate:
        model_results = [r for r in all_results if 
                        (r.get('model_type') == model_type and 
                         (r.get('model_id') == model_identifier or r.get('model_name') == model_identifier))]
        
        # Print header for this model
        model_display_name = f"{model_identifier} ({model_type})"
        logger.info(f"\nModel: {model_display_name}")
        logger.info("-" * (len(model_display_name) + 7))
        
        # Process each dataset result for this model
        for result in model_results:
            if 'prediction_distribution' in result and 'ground_truth_distribution' in result:
                dataset_name = result['dataset_name']
                pred_dist = result['prediction_distribution']['frequencies']
                gt_dist = result['ground_truth_distribution']['frequencies']
                
                logger.info(f"\nDataset: {dataset_name}")
                logger.info(f"  Ground Truth Distribution: {gt_dist}")
                logger.info(f"  Prediction Distribution:   {pred_dist}")
                
                # Calculate distribution difference
                all_labels = set(pred_dist.keys()) | set(gt_dist.keys())
                max_diff = 0
                diffs = {}
                for label in all_labels:
                    pred_freq = pred_dist.get(label, 0.0)
                    gt_freq = gt_dist.get(label, 0.0)
                    diff = abs(pred_freq - gt_freq)
                    diffs[label] = diff
                    max_diff = max(max_diff, diff)
                
                logger.info(f"  Max frequency difference: {max_diff:.4f}")
                logger.info(f"  Per-class differences: {diffs}")
    
    logger.info("\n=== End of Frequency Distribution Summary ===\n")
    
    # Log final metrics to W&B
    if args.use_wandb and WANDB_AVAILABLE:
        # Create a summary table
        model_comparison_table = wandb.Table(columns=["Model", "Type", "Average Accuracy", "Average Balanced Accuracy", "Average ROC AUC", "Datasets"])
        
        for model_summary in summary['model_summaries']:
            # Calculate average balanced accuracy if available
            avg_balanced_acc = None
            if 'dataset_balanced_accuracies' in model_summary:
                balanced_accs = [acc for acc in model_summary['dataset_balanced_accuracies'].values() if acc is not None]
                if balanced_accs:
                    avg_balanced_acc = sum(balanced_accs) / len(balanced_accs)
            
            # Calculate average ROC AUC if available
            avg_roc_auc = None
            if 'dataset_roc_aucs' in model_summary:
                roc_aucs = [auc for auc in model_summary['dataset_roc_aucs'].values() if auc is not None]
                if roc_aucs:
                    avg_roc_auc = sum(roc_aucs) / len(roc_aucs)
            
            model_comparison_table.add_data(
                model_summary['model_identifier'],
                model_summary['model_type'],
                model_summary['average_accuracy'],
                avg_balanced_acc if avg_balanced_acc is not None else None,
                avg_roc_auc if avg_roc_auc is not None else None,
                len(model_summary['dataset_accuracies'])
            )
        
        # Log the table
        wandb.log({"overall/model_comparison": model_comparison_table})
        
        # Create comparison plot for models
        try:
            import matplotlib.pyplot as plt
            
            # Group by model type
            model_types = set([ms['model_type'] for ms in summary['model_summaries']])
            
            # Create a plot for each model type
            for model_type in model_types:
                type_summaries = [ms for ms in summary['model_summaries'] if ms['model_type'] == model_type]
                
                # Sort by average accuracy
                type_summaries.sort(key=lambda x: x['average_accuracy'], reverse=True)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                model_names = [ms['model_identifier'] for ms in type_summaries]
                accuracies = [ms['average_accuracy'] for ms in type_summaries]
                
                ax.bar(range(len(model_names)), accuracies)
                ax.set_xlabel('Model')
                ax.set_ylabel('Average Accuracy')
                ax.set_title(f'{model_type.capitalize()} Models Comparison')
                ax.set_xticks(range(len(model_names)))
                ax.set_xticklabels(model_names, rotation=45, ha='right')
                
                plt.tight_layout()
                wandb.log({f"overall/{model_type}_models_comparison": wandb.Image(fig)})
                plt.close(fig)
            
            # Create an overall ranking plot with all models
            all_summaries = summary['model_summaries']
            all_summaries.sort(key=lambda x: x['average_accuracy'], reverse=True)
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            model_names = [f"{ms['model_identifier']} ({ms['model_type']})" for ms in all_summaries]
            accuracies = [ms['average_accuracy'] for ms in all_summaries]
            colors = ['#1f77b4' if ms['model_type'] == 'clam' or ms['model_type'] == 'llm' else
                     '#ff7f0e' if ms['model_type'] == 'baseline' else '#2ca02c'
                     for ms in all_summaries]
            
            ax.bar(range(len(model_names)), accuracies, color=colors)
            ax.set_xlabel('Model')
            ax.set_ylabel('Average Accuracy')
            ax.set_title(f'All Models Comparison')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            
            plt.tight_layout()
            wandb.log({"overall/all_models_comparison": wandb.Image(fig)})
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Error creating model comparison plots: {e}")
        
        # Finalize wandb run
        wandb.finish()
    
    logger.info(f"Evaluation complete! Results saved to {args.output_dir}")
    
    # Clean up GPU monitoring
    cleanup_gpu_monitoring(gpu_monitor)
    
    # Return the summary for potential scripting use
    return summary

if __name__ == "__main__":
    main()