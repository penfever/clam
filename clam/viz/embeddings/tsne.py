"""
t-SNE visualization wrapper for ContextComposer compatibility.

This provides a minimal BaseVisualization-compatible wrapper around
the existing t-SNE functions for use with multi-visualization contexts.
"""

import numpy as np
from typing import Any, Dict, Optional, List
import logging

from sklearn.manifold import TSNE
from ..base import BaseVisualization, VisualizationResult, VisualizationConfig
from ..tsne_functions import create_tsne_visualization

logger = logging.getLogger(__name__)


class TSNEVisualization(BaseVisualization):
    """
    t-SNE visualization wrapper for multi-visualization contexts.
    
    This wraps the existing CLAM t-SNE functions to be compatible with
    the BaseVisualization interface while preserving all the specialized
    functionality of the original implementation.
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
        return False  # t-SNE requires joint fitting of train+test data
    
    def _create_transformer(self, **kwargs) -> Any:
        """Create t-SNE transformer."""
        
        # Set default parameters optimized for visualization
        tsne_params = {
            'n_components': 3 if self.config.use_3d else 2,
            'perplexity': kwargs.get('perplexity', 30),
            'max_iter': kwargs.get('max_iter', 1000),
            'random_state': self.config.random_state,
            'learning_rate': kwargs.get('learning_rate', 'auto'),
            'init': kwargs.get('init', 'pca'),
            'metric': kwargs.get('metric', 'euclidean')
        }
        
        # Remove any None values
        tsne_params = {k: v for k, v in tsne_params.items() if v is not None}
        
        self.logger.info(f"Creating t-SNE with params: {tsne_params}")
        
        return TSNE(**tsne_params)
    
    def _get_default_description(self, n_samples: int, n_features: int) -> str:
        """Get default description for t-SNE."""
        dims = "3D" if self.config.use_3d else "2D"
        return (
            f"{dims} t-SNE visualization of {n_samples} samples with {n_features} features. "
            f"t-SNE preserves local neighborhood structure and reveals clusters."
        )
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Fit t-SNE and transform data.
        
        Note: For CLAM compatibility, this is designed to work with combined
        train+test data since t-SNE needs to fit all data jointly.
        """
        import time
        
        start_time = time.time()
        
        # Merge config extra_params with kwargs
        merged_kwargs = {**self.config.extra_params, **kwargs}
        
        # Create transformer
        self._transformer = self._create_transformer(**merged_kwargs)
        
        fit_start = time.time()
        
        # Fit and transform
        transformed = self._transformer.fit_transform(X)
        
        fit_time = time.time() - fit_start
        
        self._fitted = True
        
        # Store timing information
        self._last_fit_time = fit_time
        self._last_transform_time = 0.0  # Included in fit_transform
        
        self.logger.info(f"Fitted {self.method_name} in {fit_time:.2f}s")
        
        return transformed
    
    def generate_plot(
        self,
        transformed_data: np.ndarray,
        y: Optional[np.ndarray] = None,
        highlight_indices: Optional[List[int]] = None,
        test_data: Optional[np.ndarray] = None,
        highlight_test_indices: Optional[List[int]] = None,
        **kwargs
    ) -> VisualizationResult:
        """
        Generate t-SNE plot using the BaseVisualization framework.
        
        This delegates to the parent implementation for consistent plotting.
        """
        return super().generate_plot(
            transformed_data=transformed_data,
            y=y,
            highlight_indices=highlight_indices,
            test_data=test_data,
            highlight_test_indices=highlight_test_indices,
            **kwargs
        )