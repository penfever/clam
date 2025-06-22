#!/usr/bin/env python
"""
Quick test script for the new perturbation semantic axes method.
"""

import sys
import os
import numpy as np
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from clam.models.clam_tsne import ClamTsneClassifier
from sklearn.datasets import fetch_openml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_perturbation_semantic_axes():
    """Test the new perturbation method for semantic axes computation."""
    
    # Load a small dataset
    logger.info("Loading dataset (task 23: cmc)...")
    dataset = fetch_openml(data_id=23, as_frame=True, return_X_y=False)
    X = dataset.data
    y = dataset.target
    
    # Get feature names
    feature_names = list(X.columns)
    logger.info(f"Feature names: {feature_names}")
    
    # Take a small subset for testing with stratified sampling to ensure multiple classes
    from sklearn.model_selection import train_test_split
    X_small, _, y_small, _ = train_test_split(
        X, y, test_size=0.9, stratify=y, random_state=42
    )
    
    logger.info(f"Dataset shape: {X_small.shape}, classes: {len(np.unique(y_small))}")
    
    # Test traditional PCA method
    logger.info("\n=== Testing PCA method ===")
    clam_pca = ClamTsneClassifier(
        modality="tabular",
        vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        semantic_axes=True,
        semantic_axes_method="pca_loadings",
        feature_names=feature_names,
        backend="transformers",
        tsne_n_iter=250,  # Faster for testing
        enable_thinking=False
    )
    
    clam_pca.fit(X_small, y_small)
    
    if hasattr(clam_pca, 'semantic_axes_labels') and clam_pca.semantic_axes_labels:
        logger.info("PCA semantic axes:")
        for axis, label in clam_pca.semantic_axes_labels.items():
            logger.info(f"  {axis}: {label}")
    else:
        logger.warning("No PCA semantic axes computed")
    
    # Test new perturbation method
    logger.info("\n=== Testing Perturbation method ===")
    clam_pert = ClamTsneClassifier(
        modality="tabular",
        vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        semantic_axes=True,
        semantic_axes_method="perturbation",
        feature_names=feature_names,
        backend="transformers",
        tsne_n_iter=250,  # Faster for testing
        enable_thinking=False
    )
    
    clam_pert.fit(X_small, y_small)
    
    if hasattr(clam_pert, 'semantic_axes_labels') and clam_pert.semantic_axes_labels:
        logger.info("Perturbation semantic axes:")
        for axis, label in clam_pert.semantic_axes_labels.items():
            logger.info(f"  {axis}: {label}")
    else:
        logger.warning("No perturbation semantic axes computed")
    
    # Compare the two approaches
    logger.info("\n=== Comparison ===")
    if (hasattr(clam_pca, 'semantic_axes_labels') and clam_pca.semantic_axes_labels and
        hasattr(clam_pert, 'semantic_axes_labels') and clam_pert.semantic_axes_labels):
        
        logger.info("Both methods successfully computed semantic axes!")
        logger.info("PCA focuses on linear combinations of original features")
        logger.info("Perturbation measures actual impact on TabPFN embedding space")
    else:
        logger.warning("One or both methods failed to compute semantic axes")
    
    logger.info("\nTest completed!")

if __name__ == "__main__":
    test_perturbation_semantic_axes()