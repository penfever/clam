#!/usr/bin/env python3
"""
Quick test to validate the KNN parameter fix.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from clam.data.tsne_visualization import create_tsne_plot_with_knn

def test_knn_function():
    """Test that create_tsne_plot_with_knn doesn't crash with proper parameters."""
    print("Testing create_tsne_plot_with_knn function...")
    
    # Create mock data
    train_tsne = np.random.rand(50, 2)
    test_tsne = np.random.rand(1, 2)
    train_labels = np.random.randint(0, 5, 50)
    train_embeddings = np.random.rand(50, 128)
    test_embeddings = np.random.rand(1, 128)
    
    try:
        # Test with highlight_test_idx=0 (should work)
        fig, legend_text, metadata = create_tsne_plot_with_knn(
            train_tsne=train_tsne,
            test_tsne=test_tsne,
            train_labels=train_labels,
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            highlight_test_idx=0,
            k=5,
            zoom_factor=4.0,
            figsize=(10, 8)
        )
        print("✓ create_tsne_plot_with_knn works correctly with highlight_test_idx=0")
        
        # Test with highlight_test_idx=None (should also work)
        fig2, legend_text2, metadata2 = create_tsne_plot_with_knn(
            train_tsne=train_tsne,
            test_tsne=test_tsne,
            train_labels=train_labels,
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            highlight_test_idx=None,
            k=5,
            zoom_factor=4.0,
            figsize=(10, 8)
        )
        print("✓ create_tsne_plot_with_knn works correctly with highlight_test_idx=None")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    if test_knn_function():
        print("\n🎉 KNN function test passed!")
    else:
        print("\n❌ KNN function test failed!")
        sys.exit(1)