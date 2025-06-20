"""
Decision regions visualization using mlxtend.

This module provides decision boundary visualization for classifiers
using the mlxtend library's decision regions functionality.
"""

import numpy as np
from typing import Any, Dict, Optional
import logging

try:
    from mlxtend.plotting import plot_decision_regions
    import matplotlib.pyplot as plt
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

from ..base import BaseVisualization, VisualizationResult

logger = logging.getLogger(__name__)


class DecisionRegionsVisualization(BaseVisualization):
    """
    Decision regions visualization using mlxtend.
    
    This visualization shows the decision boundaries learned by a classifier
    in the embedding space, providing insight into how the model partitions
    the feature space.
    """
    
    def __init__(self, classifier=None, **kwargs):
        """
        Initialize decision regions visualization.
        
        Args:
            classifier: Trained classifier to visualize (e.g., SVM, RandomForest)
            **kwargs: Additional arguments for BaseVisualization
        """
        super().__init__(**kwargs)
        self.classifier = classifier
        self._embedding_method = None
    
    @property
    def method_name(self) -> str:
        clf_name = self.classifier.__class__.__name__ if self.classifier else "Classifier"
        return f"Decision-{clf_name}"
    
    @property
    def supports_3d(self) -> bool:
        return False  # mlxtend decision regions are typically 2D
    
    @property
    def supports_regression(self) -> bool:
        return False  # Decision regions are for classification
    
    @property
    def supports_new_data(self) -> bool:
        return True  # Can visualize decisions on new data
    
    def _create_transformer(self, **kwargs) -> Any:
        """Create embedding transformer for decision regions."""
        if not MLXTEND_AVAILABLE:
            raise ImportError(
                "mlxtend not available. Install with: pip install mlxtend"
            )
        
        # For decision regions, we need an embedding method first
        embedding_method = kwargs.get('embedding_method', 'pca')
        
        if embedding_method == 'pca':
            from sklearn.decomposition import PCA
            transformer = PCA(n_components=2, random_state=self.config.random_state)
        elif embedding_method == 'tsne':
            from sklearn.manifold import TSNE
            transformer = TSNE(n_components=2, random_state=self.config.random_state)
        elif embedding_method == 'umap':
            try:
                import umap
                transformer = umap.UMAP(n_components=2, random_state=self.config.random_state)
            except ImportError:
                logger.warning("UMAP not available, falling back to PCA")
                from sklearn.decomposition import PCA
                transformer = PCA(n_components=2, random_state=self.config.random_state)
        else:
            raise ValueError(f"Unsupported embedding method: {embedding_method}")
        
        self._embedding_method = embedding_method
        return transformer
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """Fit embedding and classifier, then transform data."""
        # First, fit the embedding
        embedded = super().fit_transform(X, y, **kwargs)
        
        # Then, fit the classifier on the embedded data
        if self.classifier is not None:
            self.classifier.fit(embedded, y)
            self.logger.info(f"Fitted {self.classifier.__class__.__name__} on {self._embedding_method} embedding")
        
        return embedded
    
    def _get_default_description(self, n_samples: int, n_features: int) -> str:
        """Get default description for decision regions."""
        clf_name = self.classifier.__class__.__name__ if self.classifier else "classifier"
        embedding_name = getattr(self, '_embedding_method', 'embedding')
        
        description = (
            f"Decision regions for {clf_name} on {embedding_name} embedding "
            f"of {n_samples} samples from {n_features} dimensions. "
            f"Shows how the classifier partitions the embedded feature space."
        )
        
        return description
    
    def generate_plot(
        self,
        transformed_data: np.ndarray,
        y: Optional[np.ndarray] = None,
        highlight_indices: Optional[list] = None,
        test_data: Optional[np.ndarray] = None,
        **kwargs
    ) -> VisualizationResult:
        """Generate decision regions plot."""
        if not MLXTEND_AVAILABLE:
            raise ImportError("mlxtend not available for decision regions")
        
        if self.classifier is None:
            raise ValueError("Classifier must be provided for decision regions visualization")
        
        if transformed_data.shape[1] != 2:
            raise ValueError("Decision regions visualization requires 2D data")
        
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot decision regions
        plot_decision_regions(
            X=transformed_data,
            y=y.astype(int),
            clf=self.classifier,
            ax=ax,
            legend=2,
            **kwargs
        )
        
        # Highlight specific points if requested
        if highlight_indices:
            ax.scatter(
                transformed_data[highlight_indices, 0],
                transformed_data[highlight_indices, 1],
                c='red', s=self.config.point_size * 2, alpha=1.0,
                marker='x', linewidths=3, label='Highlighted'
            )
        
        # Plot test data
        if test_data is not None:
            ax.scatter(
                test_data[:, 0],
                test_data[:, 1],
                c='black', s=self.config.point_size * 1.5, alpha=0.8,
                marker='^', label='Test points'
            )
        
        # Apply styling
        self._apply_plot_styling(ax, use_3d=False)
        
        if self.config.tight_layout:
            plt.tight_layout()
        
        # Convert to image
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=self.config.dpi, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        image = Image.open(img_buffer)
        plt.close(fig)
        
        # Convert to desired format
        if self.config.image_format == 'RGB' and image.mode != 'RGB':
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                image = rgb_image
            else:
                image = image.convert('RGB')
        
        # Create metadata
        metadata = {
            'classifier': self.classifier.__class__.__name__,
            'embedding_method': self._embedding_method,
            'n_classes': len(np.unique(y)) if y is not None else 0,
            'decision_boundary_type': 'regions'
        }
        
        # Create result
        result = VisualizationResult(
            image=image,
            transformed_data=transformed_data,
            description=self._get_default_description(len(transformed_data), transformed_data.shape[1]),
            method_name=self.method_name,
            config=self.config,
            highlighted_indices=highlight_indices,
            highlighted_coords=transformed_data[highlight_indices] if highlight_indices else None,
            legend_text=f"Decision regions for {self.classifier.__class__.__name__}",
            metadata=metadata
        )
        
        return result


def create_decision_regions_visualization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier,
    embedding_method: str = 'pca',
    X_test: Optional[np.ndarray] = None,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to create decision regions visualization.
    
    Args:
        X_train: Training data
        y_train: Training labels
        classifier: Classifier to visualize
        embedding_method: Method for dimensionality reduction ('pca', 'tsne', 'umap')
        X_test: Test data (optional)
        random_state: Random seed
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing visualization results
    """
    from ..base import VisualizationConfig
    
    # Create configuration
    config = VisualizationConfig(
        random_state=random_state,
        extra_params={
            'embedding_method': embedding_method,
            **kwargs
        }
    )
    
    # Create visualization
    viz = DecisionRegionsVisualization(classifier=classifier, config=config)
    
    # Fit and transform
    train_embedding = viz.fit_transform(X_train, y_train)
    test_embedding = viz.transform(X_test) if X_test is not None else None
    
    # Generate plot
    result = viz.generate_plot(
        transformed_data=train_embedding,
        y=y_train,
        test_data=test_embedding
    )
    
    return {
        'train_embedding': train_embedding,
        'test_embedding': test_embedding,
        'visualization_result': result,
        'classifier': classifier,
        'embedding_transformer': viz._transformer
    }