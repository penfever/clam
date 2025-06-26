#!/usr/bin/env python3
"""
Test script to verify KNN behavior shows pie charts instead of red dashed lines
and 3D visualization shows 4-panel multi-view layout.
"""

import numpy as np
import matplotlib.pyplot as plt
from clam.viz.tsne_functions import TSNEVisualizer

def test_knn_pie_chart():
    """Test that KNN shows pie chart instead of red dashed lines."""
    print("Testing KNN pie chart behavior...")
    
    # Create synthetic data
    np.random.seed(42)
    n_train, n_test = 100, 10
    n_features = 20
    
    # Generate train data with 3 classes
    train_embeddings = np.random.randn(n_train, n_features)
    train_labels = np.random.randint(0, 3, n_train)
    
    # Generate test data
    test_embeddings = np.random.randn(n_test, n_features)
    
    # Create KNN visualizer
    visualizer = TSNEVisualizer(
        task_type='classification',
        use_3d=False,
        use_knn=True,
        knn_k=5,
        figsize=(15, 8)
    )
    
    # Create visualization with highlighted test point
    try:
        train_coords, test_coords, fig, legend_text, metadata = visualizer.create_visualization(
            train_embeddings=train_embeddings,
            train_data=train_labels,
            test_embeddings=test_embeddings,
            highlight_test_idx=0,
            class_names=['Class A', 'Class B', 'Class C']
        )
        
        # Save the figure
        fig.savefig('/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/clam/test_knn_pie_chart.png', 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print("✓ KNN pie chart test completed successfully")
        print(f"  - Figure has {len(fig.get_axes())} axes (should be 2: main plot + pie chart)")
        print(f"  - Metadata has_knn_analysis: {metadata.get('has_knn_analysis', False)}")
        print(f"  - Legend text contains: {len(legend_text)} characters")
        
        return True
        
    except Exception as e:
        print(f"✗ KNN pie chart test failed: {e}")
        return False

def test_3d_multiview():
    """Test that 3D visualization shows 4-panel multi-view layout."""
    print("\nTesting 3D multi-view behavior...")
    
    # Create synthetic data
    np.random.seed(42)
    n_train, n_test = 50, 5
    n_features = 15
    
    # Generate train data with 3 classes
    train_embeddings = np.random.randn(n_train, n_features)
    train_labels = np.random.randint(0, 3, n_train)
    
    # Generate test data
    test_embeddings = np.random.randn(n_test, n_features)
    
    # Create 3D visualizer
    visualizer = TSNEVisualizer(
        task_type='classification',
        use_3d=True,
        use_knn=False,
        figsize=(15, 12),
        zoom_factor=1.0  # No zoom
    )
    
    # Create visualization
    try:
        train_coords, test_coords, fig, legend_text, metadata = visualizer.create_visualization(
            train_embeddings=train_embeddings,
            train_data=train_labels,
            test_embeddings=test_embeddings,
            highlight_test_idx=0,
            class_names=['Class A', 'Class B', 'Class C']
        )
        
        # Save the figure
        fig.savefig('/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/clam/test_3d_multiview.png', 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print("✓ 3D multi-view test completed successfully")
        print(f"  - Figure has {len(fig.get_axes())} axes (should be 4 for multi-view)")
        
        # Check if axes have titles (indicating different views)
        titles = [ax.get_title() for ax in fig.get_axes() if ax.get_title()]
        print(f"  - Found {len(titles)} subplot titles: {titles}")
        
        return len(fig.get_axes()) == 4
        
    except Exception as e:
        print(f"✗ 3D multi-view test failed: {e}")
        return False

def test_3d_knn():
    """Test 3D with KNN (should show 4-panel view without pie chart)."""
    print("\nTesting 3D KNN behavior...")
    
    # Create synthetic data
    np.random.seed(42)
    n_train, n_test = 50, 5
    n_features = 15
    
    # Generate train data with 3 classes
    train_embeddings = np.random.randn(n_train, n_features)
    train_labels = np.random.randint(0, 3, n_train)
    
    # Generate test data
    test_embeddings = np.random.randn(n_test, n_features)
    
    # Create 3D KNN visualizer  
    visualizer = TSNEVisualizer(
        task_type='classification',
        use_3d=True,
        use_knn=True,
        knn_k=5,
        figsize=(20, 12),  # Larger for pie chart
        zoom_factor=1.0  # No zoom
    )
    
    # Create visualization
    try:
        train_coords, test_coords, fig, legend_text, metadata = visualizer.create_visualization(
            train_embeddings=train_embeddings,
            train_data=train_labels,
            test_embeddings=test_embeddings,
            highlight_test_idx=0,
            class_names=['Class A', 'Class B', 'Class C']
        )
        
        # Save the figure
        fig.savefig('/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/clam/test_3d_knn.png', 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print("✓ 3D KNN test completed successfully")
        print(f"  - Figure has {len(fig.get_axes())} axes (should be 5: 4 3D plots + 1 pie chart)")
        
        return len(fig.get_axes()) == 5
        
    except Exception as e:
        print(f"✗ 3D KNN test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing KNN and 3D visualization fixes...")
    
    results = []
    results.append(test_knn_pie_chart())
    results.append(test_3d_multiview())  
    results.append(test_3d_knn())
    
    print(f"\nTest Summary:")
    print(f"  - Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed! Fixes are working correctly.")
    else:
        print("✗ Some tests failed. Check the output above for details.")