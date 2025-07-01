#!/usr/bin/env python
"""
Simple test script to verify zoom factor functionality.
"""

import numpy as np
import logging
from clam.models.clam_tsne import ClamTsneClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_zoom_factor():
    """Test that zoom factor is properly applied."""
    
    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 3, 100)
    X_test = np.random.randn(20, 10)
    y_test = np.random.randint(0, 3, 20)
    
    print("Testing zoom factor functionality...")
    
    # Test with zoom_factor = 1.0 (no zoom)
    print("\n=== Testing with zoom_factor=1.0 ===")
    classifier_no_zoom = ClamTsneClassifier(
        modality="tabular",
        vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        embedding_size=50,  # Small for faster testing
        tsne_perplexity=10,
        tsne_max_iter=100,  # Quick for testing
        zoom_factor=1.0,
        max_tabpfn_samples=100,
        backend="transformers",
        seed=42
    )
    
    print(f"Classifier zoom_factor: {classifier_no_zoom.zoom_factor}")
    
    # Test with zoom_factor = 4.0 (significant zoom)
    print("\n=== Testing with zoom_factor=4.0 ===")
    classifier_zoom = ClamTsneClassifier(
        modality="tabular", 
        vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        embedding_size=50,  # Small for faster testing
        tsne_perplexity=10,
        tsne_max_iter=100,  # Quick for testing
        zoom_factor=4.0,
        max_tabpfn_samples=100,
        backend="transformers",
        seed=42
    )
    
    print(f"Classifier zoom_factor: {classifier_zoom.zoom_factor}")
    
    # Test multi-viz mode
    print("\n=== Testing multi-viz mode with zoom_factor=1.0 ===")
    classifier_multi_viz = ClamTsneClassifier(
        modality="tabular",
        vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        embedding_size=50,
        tsne_perplexity=10,
        tsne_max_iter=100,
        zoom_factor=1.0,
        max_tabpfn_samples=100,
        backend="transformers",
        enable_multi_viz=True,
        seed=42
    )
    
    print(f"Multi-viz classifier zoom_factor: {classifier_multi_viz.zoom_factor}")
    
    print("\n✅ All zoom factor configurations initialized successfully!")
    print("Note: Full visualization testing requires running fit() which may take longer.")

if __name__ == "__main__":
    test_zoom_factor()