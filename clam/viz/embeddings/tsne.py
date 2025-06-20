"""
t-SNE visualization implementation.

This wraps the existing t-SNE functionality in the new visualization framework
while maintaining compatibility with the current CLAM t-SNE implementation.
"""

import numpy as np
from typing import Any, Dict
import logging

from sklearn.manifold import TSNE
from ..base import BaseVisualization, VisualizationResult

logger = logging.getLogger(__name__)


class TSNEVisualization(BaseVisualization):
    """
    t-SNE visualization implementation.
    
    t-SNE (t-Distributed Stochastic Neighbor Embedding) is excellent for
    revealing local cluster structure and non-linear patterns in data.
    This implementation wraps the existing CLAM t-SNE functionality.
    """
    
    @property
    def method_name(self) -> str:
        return "t-SNE"
    
    @property
    def supports_3d(self) -> bool:
        return True
    
    @property
    def supports_regression(self) -> bool:
        return True
    
    @property
    def supports_new_data(self) -> bool:
        return False  # t-SNE doesn't support transform on new data
    
    def _create_transformer(self, **kwargs) -> Any:
        """Create t-SNE transformer."""
        
        # Set default parameters optimized for visualization
        tsne_params = {
            'n_components': 3 if self.config.use_3d else 2,
            'perplexity': kwargs.get('perplexity', 30),
            'max_iter': kwargs.get('max_iter', 1000),
            'random_state': self.config.random_state,
            'learning_rate': kwargs.get('learning_rate', 'warn'),
            'init': kwargs.get('init', 'random'),
            'verbose': kwargs.get('verbose', 1),
            'method': kwargs.get('method', 'barnes_hut'),
            'angle': kwargs.get('angle', 0.5),
            'n_jobs': kwargs.get('n_jobs', None)
        }
        
        # Adjust perplexity for small datasets
        if hasattr(self, '_data_size'):
            effective_perplexity = min(tsne_params['perplexity'], (self._data_size - 1) // 3)
            if effective_perplexity != tsne_params['perplexity']:
                self.logger.warning(
                    f"Adjusting perplexity from {tsne_params['perplexity']} to {effective_perplexity} "
                    f"due to small dataset size"
                )
                tsne_params['perplexity'] = effective_perplexity
        
        # Remove None values and 'warn' values
        tsne_params = {k: v for k, v in tsne_params.items() if v is not None and v != 'warn'}
        
        self.logger.info(f"Creating t-SNE with parameters: {tsne_params}")
        
        return TSNE(**tsne_params)
    
    def fit_transform(self, X: np.ndarray, y=None, **kwargs) -> np.ndarray:
        """Override to store data size for perplexity adjustment."""
        self._data_size = len(X)
        return super().fit_transform(X, y, **kwargs)
    
    def _get_default_description(self, n_samples: int, n_features: int) -> str:
        """Get default description for t-SNE."""
        components = "3D" if self.config.use_3d else "2D"
        
        description = (
            f"t-SNE {components} embedding of {n_samples} samples from {n_features} dimensions. "
            f"t-SNE preserves local neighborhood structure and excels at revealing clusters "
            f"and non-linear patterns in the data."
        )
        
        # Add parameter information
        params = self.config.extra_params
        if 'perplexity' in params:
            description += f" Using perplexity {params['perplexity']} for neighborhood size."
        if 'max_iter' in params:
            description += f" Optimized for {params['max_iter']} iterations."
        
        return description
    
    def _add_quality_metrics(self, result: VisualizationResult):
        """Add t-SNE specific quality metrics."""
        if self._transformer is not None:
            # Add KL divergence (final stress)
            if hasattr(self._transformer, 'kl_divergence_'):
                result.metadata['kl_divergence'] = float(self._transformer.kl_divergence_)
            
            # Add number of iterations performed
            if hasattr(self._transformer, 'n_iter_'):
                result.metadata['n_iterations'] = int(self._transformer.n_iter_)
            
            # Add t-SNE parameters
            result.metadata['perplexity'] = self._transformer.perplexity
            result.metadata['learning_rate'] = self._transformer.learning_rate
            
            self.logger.debug(f"Added t-SNE quality metrics: {result.metadata}")


# Compatibility function with existing CLAM t-SNE interface
def create_tsne_visualization_legacy(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    use_3d: bool = False,
    **kwargs
) -> tuple:
    """
    Legacy compatibility function for existing CLAM t-SNE interface.
    
    This function maintains compatibility with the existing CLAM t-SNE
    visualization system while using the new framework internally.
    
    Returns:
        Tuple of (train_tsne, test_tsne, figure) for compatibility
    """
    from ..base import VisualizationConfig
    import matplotlib.pyplot as plt
    
    # Create configuration
    config = VisualizationConfig(
        use_3d=use_3d,
        random_state=random_state,
        extra_params={
            'perplexity': perplexity,
            'max_iter': n_iter,
            **kwargs
        }
    )
    
    # Create visualization
    viz = TSNEVisualization(config)
    
    # Combine embeddings for joint t-SNE (as in original implementation)
    combined_embeddings = np.vstack([train_embeddings, test_embeddings])
    n_train = len(train_embeddings)
    
    # Fit and transform
    combined_tsne = viz.fit_transform(combined_embeddings)
    
    # Split back
    train_tsne = combined_tsne[:n_train]
    test_tsne = combined_tsne[n_train:]
    
    # Generate plot for compatibility
    result = viz.generate_plot(
        transformed_data=train_tsne,
        y=train_labels,
        test_data=test_tsne
    )
    
    # Create a dummy figure for compatibility
    fig = plt.figure()
    plt.close(fig)  # Close immediately since we return the PIL image in result
    
    return train_tsne, test_tsne, fig