"""
Unit tests for the refactored t-SNE visualization classes.

Tests the core functionality of the new class-based architecture to ensure
it works correctly and maintains compatibility with the original functions.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clam.viz.tsne_refactored import (
    TSNEGenerator,
    BaseTSNEPlotter, 
    ClassificationTSNEPlotter,
    RegressionTSNEPlotter,
    KNNMixin,
    TSNEVisualizer
)


class TestTSNEGenerator:
    """Test the core t-SNE coordinate generation."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = TSNEGenerator(perplexity=20, max_iter=500, random_state=123)
        assert generator.perplexity == 20
        assert generator.max_iter == 500
        assert generator.random_state == 123
        
    def test_fit_transform_2d(self):
        """Test 2D coordinate generation."""
        # Create dummy data
        np.random.seed(42)
        train_embeddings = np.random.randn(50, 10)
        test_embeddings = np.random.randn(10, 10)
        
        generator = TSNEGenerator(perplexity=5, max_iter=250, random_state=42)
        train_coords, test_coords = generator.fit_transform(train_embeddings, test_embeddings, n_components=2)
        
        # Check shapes
        assert train_coords.shape == (50, 2)
        assert test_coords.shape == (10, 2)
        
        # Check that coordinates are not all zeros (t-SNE actually ran)
        assert not np.allclose(train_coords, 0)
        assert not np.allclose(test_coords, 0)
        
    def test_fit_transform_3d(self):
        """Test 3D coordinate generation."""
        np.random.seed(42)
        train_embeddings = np.random.randn(30, 5)
        test_embeddings = np.random.randn(5, 5)
        
        generator = TSNEGenerator(perplexity=3, max_iter=250, random_state=42)
        train_coords, test_coords = generator.fit_transform(train_embeddings, test_embeddings, n_components=3)
        
        # Check shapes
        assert train_coords.shape == (30, 3)
        assert test_coords.shape == (5, 3)
        
    def test_perplexity_adjustment(self):
        """Test automatic perplexity adjustment for small datasets."""
        # Very small dataset that should trigger perplexity adjustment
        np.random.seed(42)
        train_embeddings = np.random.randn(10, 3)
        test_embeddings = np.random.randn(3, 3)
        
        generator = TSNEGenerator(perplexity=30, max_iter=250, random_state=42)  # High perplexity
        
        with patch.object(generator.logger, 'warning') as mock_warning:
            train_coords, test_coords = generator.fit_transform(train_embeddings, test_embeddings)
            
            # Should have warned about perplexity adjustment
            mock_warning.assert_called_once()
            call_args = mock_warning.call_args[0][0]
            assert "Adjusting perplexity from 30 to" in call_args


class TestClassificationTSNEPlotter:
    """Test classification-specific plotting."""
    
    def test_init(self):
        """Test plotter initialization."""
        plotter = ClassificationTSNEPlotter(figsize=(12, 10), zoom_factor=3.0, use_3d=True)
        assert plotter.figsize == (12, 10)
        assert plotter.zoom_factor == 3.0
        assert plotter.use_3d == True
        
    def test_create_plot_2d(self):
        """Test 2D classification plot creation."""
        # Create dummy data
        np.random.seed(42)
        train_coords = np.random.randn(30, 2)
        train_labels = np.random.choice([0, 1, 2], 30)
        test_coords = np.random.randn(5, 2)
        
        plotter = ClassificationTSNEPlotter(use_3d=False)
        fig, legend_text, metadata = plotter.create_plot(
            train_coords, train_labels, test_coords, highlight_test_idx=0
        )
        
        # Check return types
        assert isinstance(fig, plt.Figure)
        assert isinstance(legend_text, str)
        assert isinstance(metadata, dict)
        
        # Check metadata structure
        assert 'visible_classes' in metadata
        assert 'plot_type' in metadata
        
        plt.close(fig)  # Clean up
        
    def test_create_plot_3d(self):
        """Test 3D classification plot creation."""
        np.random.seed(42)
        train_coords = np.random.randn(20, 3)
        train_labels = np.random.choice([0, 1], 20)
        test_coords = np.random.randn(3, 3)
        
        plotter = ClassificationTSNEPlotter(use_3d=True)
        fig, legend_text, metadata = plotter.create_plot(
            train_coords, train_labels, test_coords
        )
        
        # Check that we got a 3D plot
        assert len(fig.get_axes()) == 1
        ax = fig.get_axes()[0]
        assert hasattr(ax, 'zaxis')  # 3D axis has zaxis
        
        plt.close(fig)
        

class TestRegressionTSNEPlotter:
    """Test regression-specific plotting."""
    
    def test_create_plot_2d(self):
        """Test 2D regression plot creation."""
        np.random.seed(42)
        train_coords = np.random.randn(25, 2)
        train_targets = np.random.randn(25)  # Continuous targets
        test_coords = np.random.randn(4, 2)
        
        plotter = RegressionTSNEPlotter(use_3d=False)
        fig, legend_text, metadata = plotter.create_plot(
            train_coords, train_targets, test_coords, highlight_test_idx=1
        )
        
        # Check metadata for regression
        assert metadata['plot_type'] == 'regression'
        assert metadata['visible_classes'] == []  # No classes in regression
        assert 'target_range' in metadata
        assert len(metadata['target_range']) == 2
        
        # Check that legend mentions regression
        assert 'regression' in legend_text.lower()
        
        plt.close(fig)


class TestKNNMixin:
    """Test KNN functionality."""
    
    def test_knn_analysis(self):
        """Test KNN analysis computation."""
        # Create a simple test case where KNN is predictable
        np.random.seed(42)
        training_embeddings = np.array([
            [0, 0], [1, 0], [0, 1],  # Close to origin
            [10, 10], [11, 10], [10, 11]  # Far from origin
        ])
        training_labels = np.array([0, 0, 0, 1, 1, 1])
        query_point = np.array([0.5, 0.5])  # Should be closest to first 3 points
        
        # Create a class that has the KNN functionality
        class TestKNNPlotter(KNNMixin, ClassificationTSNEPlotter):
            pass
            
        plotter = TestKNNPlotter(knn_k=3)
        knn_info = plotter._compute_knn_analysis(query_point, training_embeddings, training_labels, k=3)
        
        # Check that we found the right neighbors (indices 0, 1, 2)
        assert len(knn_info['indices']) == 3
        assert len(knn_info['distances']) == 3
        assert len(knn_info['labels']) == 3
        
        # The nearest neighbors should be from class 0
        assert all(label == 0 for label in knn_info['labels'])


class TestTSNEVisualizer:
    """Test the main unified interface."""
    
    def test_init_classification(self):
        """Test initialization for classification."""
        visualizer = TSNEVisualizer(
            task_type='classification',
            use_3d=True,
            use_knn=True,
            knn_k=7,
            perplexity=25
        )
        
        assert visualizer.task_type == 'classification'
        assert visualizer.use_3d == True
        assert visualizer.use_knn == True
        assert visualizer.generator.perplexity == 25
        
    def test_init_regression(self):
        """Test initialization for regression."""
        visualizer = TSNEVisualizer(task_type='regression', use_3d=False, use_knn=False)
        
        assert visualizer.task_type == 'regression'
        assert visualizer.use_3d == False
        assert visualizer.use_knn == False
        
    def test_invalid_task_type(self):
        """Test error handling for invalid task type."""
        with pytest.raises(ValueError, match="Unknown task_type"):
            TSNEVisualizer(task_type='invalid')
            
    def test_create_visualization_classification(self):
        """Test full visualization creation for classification."""
        np.random.seed(42)
        train_embeddings = np.random.randn(40, 8)
        train_labels = np.random.choice([0, 1, 2], 40)
        test_embeddings = np.random.randn(8, 8)
        
        visualizer = TSNEVisualizer(
            task_type='classification',
            use_3d=False,
            use_knn=False,
            perplexity=5,
            max_iter=250
        )
        
        result = visualizer.create_visualization(
            train_embeddings, train_labels, test_embeddings, highlight_test_idx=2
        )
        
        train_coords, test_coords, fig, legend_text, metadata = result
        
        # Check all return values
        assert train_coords.shape == (40, 2)
        assert test_coords.shape == (8, 2)
        assert isinstance(fig, plt.Figure)
        assert isinstance(legend_text, str)
        assert isinstance(metadata, dict)
        assert 'visible_classes' in metadata
        
        plt.close(fig)
        
    def test_create_visualization_regression(self):
        """Test full visualization creation for regression."""
        np.random.seed(42)
        train_embeddings = np.random.randn(30, 5)
        train_targets = np.random.randn(30)
        test_embeddings = np.random.randn(6, 5)
        
        visualizer = TSNEVisualizer(
            task_type='regression',
            use_3d=False,
            perplexity=5,
            max_iter=250
        )
        
        result = visualizer.create_visualization(
            train_embeddings, train_targets, test_embeddings
        )
        
        train_coords, test_coords, fig, legend_text, metadata = result
        
        # Check regression-specific metadata
        assert metadata['plot_type'] == 'regression'
        assert metadata['visible_classes'] == []
        assert 'target_range' in metadata
        
        plt.close(fig)
        
    def test_create_simple_visualization(self):
        """Test simplified interface for backward compatibility."""
        np.random.seed(42)
        train_embeddings = np.random.randn(20, 4)
        train_labels = np.random.choice([0, 1], 20)
        test_embeddings = np.random.randn(4, 4)
        
        visualizer = TSNEVisualizer(perplexity=3, max_iter=100)
        train_coords, test_coords, fig = visualizer.create_simple_visualization(
            train_embeddings, train_labels, test_embeddings
        )
        
        # Check simplified return values
        assert train_coords.shape == (20, 2)
        assert test_coords.shape == (4, 2)
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)


if __name__ == "__main__":
    # Run a simple test to verify everything works
    print("Testing TSNEGenerator...")
    generator = TSNEGenerator(perplexity=5, max_iter=250)  # Fixed: min 250 iterations
    train_emb = np.random.randn(20, 5)
    test_emb = np.random.randn(5, 5)
    train_coords, test_coords = generator.fit_transform(train_emb, test_emb)
    print(f"✓ Generated coordinates: train {train_coords.shape}, test {test_coords.shape}")
    
    print("Testing TSNEVisualizer...")
    visualizer = TSNEVisualizer(task_type='classification', perplexity=3, max_iter=250)  # Fixed: min 250 iterations
    train_labels = np.random.choice([0, 1], 20)
    result = visualizer.create_visualization(train_emb, train_labels, test_emb)
    print(f"✓ Created visualization with {len(result)} return values")
    plt.close(result[2])  # Close the figure
    
    print("All tests passed! 🎉")