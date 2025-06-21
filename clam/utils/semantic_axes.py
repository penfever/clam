"""
Semantic axes computation for visualizations.

This module provides utilities for computing factor weightings of named features
to improve visualization legends by labeling the semantic factors influencing them.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

from .metadata_loader import DatasetMetadata, ColumnMetadata

logger = logging.getLogger(__name__)


class SemanticAxesComputer:
    """Computes semantic interpretations of dimensionality reduction axes."""
    
    def __init__(self, 
                 method: str = "pca_loadings",
                 top_k_features: int = 3,
                 min_loading_threshold: float = 0.1):
        """
        Initialize semantic axes computer.
        
        Args:
            method: Method for computing semantic axes ("pca_loadings", "feature_importance")
            top_k_features: Number of top features to include in axis labels
            min_loading_threshold: Minimum loading/importance to consider significant
        """
        self.method = method
        self.top_k_features = top_k_features
        self.min_loading_threshold = min_loading_threshold
        
    def compute_semantic_axes(self,
                             embeddings: np.ndarray,
                             reduced_coords: np.ndarray,
                             labels: np.ndarray,
                             feature_names: Optional[List[str]] = None,
                             metadata: Optional[DatasetMetadata] = None) -> Dict[str, str]:
        """
        Compute semantic interpretations for dimensionality reduction axes.
        
        Args:
            embeddings: Original high-dimensional embeddings [n_samples, n_features]
            reduced_coords: Low-dimensional coordinates [n_samples, n_dims]
            labels: Class labels [n_samples]
            feature_names: Names of original features
            metadata: Dataset metadata with feature descriptions
            
        Returns:
            Dictionary mapping axis names (e.g., "X", "Y", "Z") to semantic descriptions
        """
        if feature_names is None and metadata is None:
            logger.warning("No feature names or metadata provided, cannot compute semantic axes")
            return {}
            
        n_dims = reduced_coords.shape[1]
        axis_names = ["X", "Y", "Z"][:n_dims]
        
        # Get feature names and descriptions
        if metadata is not None:
            feature_names = [col.name for col in metadata.columns]
            feature_descriptions = {col.name: col.semantic_description for col in metadata.columns}
        else:
            feature_descriptions = {name: name for name in feature_names}
            
        semantic_axes = {}
        
        if self.method == "pca_loadings":
            semantic_axes = self._compute_pca_based_axes(
                embeddings, reduced_coords, feature_names, feature_descriptions, axis_names
            )
        elif self.method == "feature_importance":
            semantic_axes = self._compute_importance_based_axes(
                embeddings, reduced_coords, labels, feature_names, feature_descriptions, axis_names
            )
        else:
            logger.warning(f"Unknown semantic axes method: {self.method}")
            
        return semantic_axes
    
    def _compute_pca_based_axes(self,
                               embeddings: np.ndarray,
                               reduced_coords: np.ndarray,
                               feature_names: List[str],
                               feature_descriptions: Dict[str, str],
                               axis_names: List[str]) -> Dict[str, str]:
        """Compute semantic axes using PCA loadings on original embeddings."""
        try:
            # Standardize embeddings
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            # Apply PCA to match the dimensionality of reduced coordinates
            n_components = len(axis_names)
            pca = PCA(n_components=n_components)
            pca.fit(embeddings_scaled)
            
            semantic_axes = {}
            
            for i, axis_name in enumerate(axis_names):
                # Get loadings (principal component coefficients)
                loadings = pca.components_[i]
                
                # Find top contributing features
                top_indices = np.argsort(np.abs(loadings))[-self.top_k_features:][::-1]
                top_features = []
                
                for idx in top_indices:
                    if idx < len(feature_names) and np.abs(loadings[idx]) >= self.min_loading_threshold:
                        feature_name = feature_names[idx]
                        loading_value = loadings[idx]
                        direction = "+" if loading_value > 0 else "-"
                        
                        # Use semantic description if available, otherwise feature name
                        description = feature_descriptions.get(feature_name, feature_name)
                        
                        # Truncate long descriptions
                        if len(description) > 40:
                            description = description[:37] + "..."
                            
                        top_features.append(f"{direction}{description}")
                
                if top_features:
                    variance_explained = pca.explained_variance_ratio_[i] * 100
                    axis_description = f"{axis_name}-axis ({variance_explained:.1f}% var): {', '.join(top_features[:2])}"
                    semantic_axes[axis_name] = axis_description
                else:
                    semantic_axes[axis_name] = f"{axis_name}-axis: Mixed factors"
                    
            return semantic_axes
            
        except Exception as e:
            logger.error(f"Error computing PCA-based semantic axes: {e}")
            return {name: f"{name}-axis" for name in axis_names}
    
    def _compute_importance_based_axes(self,
                                      embeddings: np.ndarray,
                                      reduced_coords: np.ndarray,
                                      labels: np.ndarray,
                                      feature_names: List[str],
                                      feature_descriptions: Dict[str, str],
                                      axis_names: List[str]) -> Dict[str, str]:
        """Compute semantic axes using feature importance for predicting axis coordinates."""
        try:
            semantic_axes = {}
            
            for i, axis_name in enumerate(axis_names):
                if i >= reduced_coords.shape[1]:
                    break
                    
                # Use feature selection to find features that best predict this axis coordinate
                axis_coords = reduced_coords[:, i]
                
                # Discretize coordinates into bins for classification-based feature selection
                n_bins = min(5, len(np.unique(labels)))
                axis_bins = pd.cut(axis_coords, bins=n_bins, labels=False)
                
                # Select top features
                selector = SelectKBest(score_func=f_classif, k=min(self.top_k_features * 2, embeddings.shape[1]))
                selector.fit(embeddings, axis_bins)
                
                # Get feature scores
                feature_scores = selector.scores_
                top_indices = np.argsort(feature_scores)[-self.top_k_features:][::-1]
                
                top_features = []
                for idx in top_indices:
                    if idx < len(feature_names) and feature_scores[idx] >= self.min_loading_threshold:
                        feature_name = feature_names[idx]
                        description = feature_descriptions.get(feature_name, feature_name)
                        
                        # Truncate long descriptions
                        if len(description) > 40:
                            description = description[:37] + "..."
                            
                        top_features.append(description)
                
                if top_features:
                    axis_description = f"{axis_name}-axis: {', '.join(top_features[:2])}"
                    semantic_axes[axis_name] = axis_description
                else:
                    semantic_axes[axis_name] = f"{axis_name}-axis: Mixed factors"
                    
            return semantic_axes
            
        except Exception as e:
            logger.error(f"Error computing importance-based semantic axes: {e}")
            return {name: f"{name}-axis" for name in axis_names}


def create_semantic_axis_legend(semantic_axes: Dict[str, str],
                                figsize: Tuple[float, float] = (8, 6)) -> str:
    """
    Create a legend text describing semantic axes.
    
    Args:
        semantic_axes: Dictionary mapping axis names to semantic descriptions
        figsize: Figure size for context
        
    Returns:
        Formatted legend text for inclusion in VLM prompts
    """
    if not semantic_axes:
        return ""
        
    legend_lines = []
    legend_lines.append("Semantic Axis Interpretation:")
    
    for axis_name in ["X", "Y", "Z"]:
        if axis_name in semantic_axes:
            legend_lines.append(f"• {semantic_axes[axis_name]}")
            
    legend_text = "\n".join(legend_lines)
    
    return legend_text


def enhance_visualization_with_semantic_axes(embeddings: np.ndarray,
                                           reduced_coords: np.ndarray,
                                           labels: np.ndarray,
                                           metadata: Optional[DatasetMetadata] = None,
                                           feature_names: Optional[List[str]] = None,
                                           method: str = "pca_loadings") -> str:
    """
    Enhanced convenience function to compute and format semantic axes.
    
    Args:
        embeddings: Original high-dimensional embeddings
        reduced_coords: Reduced dimensionality coordinates  
        labels: Class labels
        metadata: Dataset metadata
        feature_names: Feature names (if metadata not available)
        method: Method for computing semantic axes
        
    Returns:
        Formatted semantic axes legend text
    """
    computer = SemanticAxesComputer(method=method)
    semantic_axes = computer.compute_semantic_axes(
        embeddings, reduced_coords, labels, feature_names, metadata
    )
    
    return create_semantic_axis_legend(semantic_axes)