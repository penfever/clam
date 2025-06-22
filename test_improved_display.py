#!/usr/bin/env python
"""
Quick test of the improved semantic axes display system.
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

def test_improved_display():
    """Test the improved display system for semantic axes."""
    
    # Load a small dataset
    logger.info("Loading dataset (task 23: cmc)...")
    dataset = fetch_openml(data_id=23, as_frame=True, return_X_y=False)
    X = dataset.data
    y = dataset.target
    
    # Take a small subset
    from sklearn.model_selection import train_test_split
    X_small, _, y_small, _ = train_test_split(
        X, y, test_size=0.9, stratify=y, random_state=42
    )
    
    logger.info(f"Dataset shape: {X_small.shape}, classes: {len(np.unique(y_small))}")
    
    # Test the improved PCA method with semantic axes
    logger.info("Testing improved semantic axes display...")
    clam_pca = ClamTsneClassifier(
        modality="tabular",
        vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        semantic_axes=True,
        semantic_axes_method="pca_loadings",
        feature_names=list(X.columns),
        backend="transformers",
        tsne_n_iter=250,
        enable_thinking=False,
        use_semantic_names=True,
        load_semantic_from_cc18=True
    )
    
    # Fit the model
    clam_pca.fit(X_small, y_small)
    
    # The model already has visualizations from fit()
    # Just check if semantic axes were computed correctly
    logger.info("Model fitted successfully")
    
    logger.info("Visualization saved to ./test_improved_display_output/")
    logger.info("Check the visualization to see:")
    logger.info("1. Compact axis labels (not overlapping)")
    logger.info("2. Bottom legend with full semantic information")
    logger.info("3. (+) and (-) prefixes for feature directions")
    
    if hasattr(clam_pca, 'semantic_axes_labels') and clam_pca.semantic_axes_labels:
        logger.info("Semantic axes computed:")
        for axis, label in clam_pca.semantic_axes_labels.items():
            logger.info(f"  {axis}: {label}")
    
    logger.info("Test completed!")

if __name__ == "__main__":
    test_improved_display()