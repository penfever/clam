"""
Semantic axes computation for visualizations.

This module provides utilities for computing factor weightings of named features
to improve visualization legends by labeling the semantic factors influencing them.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
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
                 min_loading_threshold: float = 0.05,
                 perturbation_samples: int = 50,
                 perturbation_strength: float = 0.1):
        """
        Initialize semantic axes computer.
        
        Args:
            method: Method for computing semantic axes ("pca_loadings", "feature_importance", "perturbation")
            top_k_features: Number of top features to include in axis labels
            min_loading_threshold: Minimum loading/importance to consider significant
            perturbation_samples: Number of perturbation samples for perturbation method
            perturbation_strength: Strength of perturbations as fraction of feature std
        """
        self.method = method
        self.top_k_features = top_k_features
        self.min_loading_threshold = min_loading_threshold
        self.perturbation_samples = perturbation_samples
        self.perturbation_strength = perturbation_strength
        
    def compute_semantic_axes(self,
                             embeddings: np.ndarray,
                             reduced_coords: np.ndarray,
                             labels: np.ndarray,
                             feature_names: Optional[List[str]] = None,
                             metadata: Optional[DatasetMetadata] = None,
                             original_features: Optional[np.ndarray] = None,
                             embedding_model: Optional[Any] = None,
                             reduction_func: Optional[callable] = None) -> Dict[str, str]:
        """
        Compute semantic interpretations for dimensionality reduction axes.
        
        Args:
            embeddings: Original high-dimensional embeddings [n_samples, n_features]
            reduced_coords: Low-dimensional coordinates [n_samples, n_dims]
            labels: Class labels [n_samples]
            feature_names: Names of original features
            metadata: Dataset metadata with feature descriptions
            original_features: Original feature matrix before embedding (for perturbation method)
            embedding_model: Model used to generate embeddings (for perturbation method)
            reduction_func: Function to apply dimensionality reduction (for perturbation method)
            
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
        elif self.method == "perturbation":
            if original_features is None or embedding_model is None or reduction_func is None:
                logger.warning("Perturbation method requires original_features, embedding_model, and reduction_func")
                # Fallback to PCA method
                semantic_axes = self._compute_pca_based_axes(
                    embeddings, reduced_coords, feature_names, feature_descriptions, axis_names
                )
            else:
                semantic_axes = self._compute_perturbation_based_axes(
                    original_features, reduced_coords, feature_names, feature_descriptions, 
                    axis_names, embedding_model, reduction_func
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
            
            # Debug logging
            logger.info(f"PCA semantic axes: embeddings shape {embeddings.shape}, feature_names count {len(feature_names) if feature_names else 0}")
            logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
            
            for i, axis_name in enumerate(axis_names):
                # Get loadings (principal component coefficients)
                loadings = pca.components_[i]
                variance_explained = pca.explained_variance_ratio_[i] * 100
                
                # Find top contributing features
                top_indices = np.argsort(np.abs(loadings))[-self.top_k_features:][::-1]
                top_features = []
                
                # Debug logging for this axis
                max_loading = np.max(np.abs(loadings))
                logger.info(f"Axis {axis_name}: max loading {max_loading:.3f}, threshold {self.min_loading_threshold}")
                
                for idx in top_indices:
                    if idx < len(feature_names):
                        loading_value = loadings[idx]
                        if np.abs(loading_value) >= self.min_loading_threshold:
                            feature_name = feature_names[idx]
                            direction = "+" if loading_value > 0 else "-"
                            
                            # Use semantic description if available, otherwise feature name
                            description = feature_descriptions.get(feature_name, feature_name)
                            
                            # Truncate long descriptions
                            if len(description) > 40:
                                description = description[:37] + "..."
                                
                            top_features.append(f"{direction}{description}")
                
                if top_features:
                    axis_description = f"{axis_name}-axis ({variance_explained:.1f}% var): {', '.join(top_features[:2])}"
                    semantic_axes[axis_name] = axis_description
                else:
                    # Fallback: show variance even without clear semantic factors
                    # Use top features regardless of threshold for fallback
                    fallback_features = []
                    for idx in top_indices[:2]:  # Top 2 features
                        if idx < len(feature_names):
                            feature_name = feature_names[idx]
                            loading_value = loadings[idx]
                            direction = "+" if loading_value > 0 else "-"
                            description = feature_descriptions.get(feature_name, feature_name)
                            if len(description) > 30:
                                description = description[:27] + "..."
                            fallback_features.append(f"{direction}{description}")
                    
                    if fallback_features:
                        axis_description = f"{axis_name}-axis ({variance_explained:.1f}% var): {', '.join(fallback_features)} (weak)"
                    else:
                        axis_description = f"{axis_name}-axis ({variance_explained:.1f}% var): Mixed factors"
                    
                    semantic_axes[axis_name] = axis_description
                    
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

    def _compute_perturbation_based_axes(self,
                                        original_features: np.ndarray,
                                        baseline_reduced: np.ndarray,
                                        feature_names: List[str],
                                        feature_descriptions: Dict[str, str],
                                        axis_names: List[str],
                                        embedding_model: Any,
                                        reduction_func: callable) -> Dict[str, str]:
        """
        Compute semantic axes using perturbation-based sensitivity analysis.
        
        For each feature, measure how perturbing it affects the reduced coordinates.
        This works well with TabPFN embeddings where direct feature-factor relationships are lost.
        """
        try:
            logger.info(f"Computing perturbation-based semantic axes with {self.perturbation_samples} samples")
            
            n_features = original_features.shape[1]
            n_dims = len(axis_names)
            
            # Store sensitivities: [n_features, n_dims]
            feature_sensitivities = np.zeros((n_features, n_dims))
            
            # Compute standard deviations for perturbation scaling
            feature_stds = np.std(original_features, axis=0)
            
            for feature_idx in range(n_features):
                logger.debug(f"Processing feature {feature_idx+1}/{n_features}: {feature_names[feature_idx] if feature_idx < len(feature_names) else f'feature_{feature_idx}'}")
                
                perturbation_shifts = []
                
                for sample_idx in range(self.perturbation_samples):
                    # Create perturbed version of the data
                    X_perturbed = original_features.copy()
                    
                    # Add noise to this feature
                    feature_std = feature_stds[feature_idx]
                    if feature_std > 0:  # Avoid division by zero for constant features
                        noise = np.random.normal(0, self.perturbation_strength * feature_std, 
                                               original_features.shape[0])
                        X_perturbed[:, feature_idx] += noise
                    
                    try:
                        # Get perturbed embeddings and reduced coordinates
                        perturbed_embeddings = embedding_model.transform(X_perturbed)
                        perturbed_reduced = reduction_func(perturbed_embeddings)
                        
                        # Ensure same shape as baseline
                        if perturbed_reduced.shape != baseline_reduced.shape:
                            logger.warning(f"Shape mismatch in perturbation: {perturbed_reduced.shape} vs {baseline_reduced.shape}")
                            continue
                            
                        # Measure shift in reduced coordinates
                        coordinate_shift = np.mean(np.abs(perturbed_reduced - baseline_reduced), axis=0)
                        perturbation_shifts.append(coordinate_shift)
                        
                    except Exception as e:
                        logger.debug(f"Error in perturbation sample {sample_idx} for feature {feature_idx}: {e}")
                        continue
                
                if perturbation_shifts:
                    # Average sensitivity across all perturbation samples
                    feature_sensitivities[feature_idx] = np.mean(perturbation_shifts, axis=0)
                else:
                    logger.warning(f"No valid perturbations for feature {feature_idx}")
            
            # Create semantic axes from sensitivities
            semantic_axes = {}
            
            for dim_idx, axis_name in enumerate(axis_names):
                # Get sensitivities for this dimension
                dim_sensitivities = feature_sensitivities[:, dim_idx]
                
                # Find top contributing features
                top_indices = np.argsort(dim_sensitivities)[-self.top_k_features:][::-1]
                
                # Filter by threshold
                significant_features = []
                max_sensitivity = np.max(dim_sensitivities) if len(dim_sensitivities) > 0 else 0
                
                for idx in top_indices:
                    sensitivity = dim_sensitivities[idx]
                    if max_sensitivity > 0 and sensitivity >= self.min_loading_threshold * max_sensitivity:
                        if idx < len(feature_names):
                            feature_name = feature_names[idx]
                            description = feature_descriptions.get(feature_name, feature_name)
                            
                            # Truncate long descriptions
                            if len(description) > 40:
                                description = description[:37] + "..."
                                
                            # Include sensitivity score in description
                            sensitivity_pct = (sensitivity / max_sensitivity * 100) if max_sensitivity > 0 else 0
                            significant_features.append(f"{description} ({sensitivity_pct:.0f}%)")
                
                # Create axis description
                if significant_features:
                    features_str = ", ".join(significant_features[:2])  # Top 2 features
                    axis_description = f"{axis_name}-axis: {features_str}"
                else:
                    # Fallback: show top features regardless of threshold
                    fallback_features = []
                    for idx in top_indices[:2]:
                        if idx < len(feature_names):
                            feature_name = feature_names[idx]
                            description = feature_descriptions.get(feature_name, feature_name)
                            if len(description) > 30:
                                description = description[:27] + "..."
                            fallback_features.append(description)
                    
                    if fallback_features:
                        axis_description = f"{axis_name}-axis: {', '.join(fallback_features)} (weak)"
                    else:
                        axis_description = f"{axis_name}-axis: Mixed factors"
                
                semantic_axes[axis_name] = axis_description
                
                # Debug logging
                logger.debug(f"Axis {axis_name}: top sensitivities = {dim_sensitivities[top_indices[:3]]}")
            
            return semantic_axes
            
        except Exception as e:
            logger.error(f"Error computing perturbation-based semantic axes: {e}")
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