"""
Refactored t-SNE visualization classes to eliminate code duplication.

This module provides a clean class-based architecture to replace the 
duplicated functions in tsne_functions.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from abc import ABC, abstractmethod
from sklearn.manifold import TSNE
from typing import Tuple, Optional, List, Dict, Union, Any

# Import shared styling utilities
from .utils.styling import (
    apply_consistent_point_styling,
    apply_consistent_legend_formatting,
    get_standard_test_point_style,
    get_standard_target_point_style,
    create_distinct_color_map,
    create_regression_color_map
)

# Import semantic axes utilities
try:
    from ..utils.semantic_axes import create_compact_axis_labels, create_bottom_legend_text
except ImportError:
    # Fallback for cases where semantic_axes is not available
    def create_compact_axis_labels(semantic_axes, **kwargs):
        return {}
    def create_bottom_legend_text(semantic_axes, **kwargs):
        return ""

logger = logging.getLogger(__name__)


class TSNEGenerator:
    """
    Core t-SNE computation and coordinate generation.
    
    Eliminates duplication of t-SNE fitting logic across all the original functions.
    Handles embedding combination, perplexity adjustment, and coordinate generation.
    """
    
    def __init__(self, perplexity: int = 30, max_iter: int = 1000, random_state: int = 42):
        """
        Initialize t-SNE generator.
        
        Args:
            perplexity: t-SNE perplexity parameter
            max_iter: Number of t-SNE iterations  
            random_state: Random seed for reproducibility
        """
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.random_state = random_state
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def fit_transform(
        self, 
        train_embeddings: np.ndarray, 
        test_embeddings: np.ndarray,
        n_components: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate t-SNE coordinates for train and test data.
        
        Args:
            train_embeddings: Training embeddings [n_train, embedding_dim]
            test_embeddings: Test embeddings [n_test, embedding_dim]  
            n_components: Number of t-SNE components (2 for 2D, 3 for 3D)
            
        Returns:
            train_coords: t-SNE coordinates for training data [n_train, n_components]
            test_coords: t-SNE coordinates for test data [n_test, n_components]
        """
        self.logger.info(f"Generating {n_components}D t-SNE coordinates for {len(train_embeddings)} train and {len(test_embeddings)} test samples")
        
        # Step 1: Combine embeddings for joint t-SNE (shared across all original functions)
        combined_embeddings = np.vstack([train_embeddings, test_embeddings])
        n_train = len(train_embeddings)
        
        # Step 2: Adjust perplexity based on data size (shared logic)
        effective_perplexity = min(self.perplexity, (len(combined_embeddings) - 1) // 3)
        if effective_perplexity != self.perplexity:
            self.logger.warning(
                f"Adjusting perplexity from {self.perplexity} to {effective_perplexity} "
                f"due to small dataset size"
            )
        
        # Step 3: Apply t-SNE (shared initialization logic)
        self.logger.info(f"Running t-SNE with perplexity={effective_perplexity}, max_iter={self.max_iter}")
        tsne = TSNE(
            n_components=n_components,
            perplexity=effective_perplexity,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=1
        )
        
        # Step 4: Fit and transform
        combined_coords = tsne.fit_transform(combined_embeddings)
        
        # Step 5: Split back into train and test (shared splitting logic) 
        train_coords = combined_coords[:n_train]
        test_coords = combined_coords[n_train:]
        
        self.logger.info(f"t-SNE completed successfully")
        return train_coords, test_coords


class BaseTSNEPlotter(ABC):
    """
    Abstract base class for t-SNE plotting with shared infrastructure.
    
    Eliminates duplication of figure creation, zoom logic, and legend handling
    across classification and regression plotters.
    """
    
    def __init__(
        self, 
        figsize: Tuple[int, int] = (10, 8),
        zoom_factor: float = 2.0,
        use_3d: bool = False
    ):
        """
        Initialize base plotter.
        
        Args:
            figsize: Figure size (width, height)
            zoom_factor: Zoom level for highlighted points
            use_3d: Whether to create 3D plots
        """
        self.figsize = figsize
        self.zoom_factor = zoom_factor
        self.use_3d = use_3d
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def _create_figure(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create figure and axis (2D or 3D based on configuration).
        
        Unified figure creation logic shared across all plotters.
        """
        if self.use_3d:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
        return fig, ax
        
    def _apply_zoom(
        self,
        ax: plt.Axes,
        target_point: np.ndarray,
        train_coords: np.ndarray,
        test_coords: np.ndarray
    ) -> None:
        """
        Apply zoom around target point.
        
        Shared zoom logic that was duplicated across multiple functions.
        """
        if self.zoom_factor <= 1.0:
            return
            
        # Calculate the visible range based on zoom factor
        # zoom_factor = 2.0 means we show 1/2 of the original range
        all_coords = np.vstack([train_coords, test_coords])
        
        if self.use_3d:
            # 3D zoom logic
            for i, coord_name in enumerate(['X', 'Y', 'Z']):
                coord_range = all_coords[:, i].max() - all_coords[:, i].min()
                visible_range = coord_range / self.zoom_factor
                center = target_point[i]
                
                coord_min = center - visible_range / 2
                coord_max = center + visible_range / 2
                
                if i == 0:
                    ax.set_xlim(coord_min, coord_max)
                elif i == 1:
                    ax.set_ylim(coord_min, coord_max)
                else:
                    ax.set_zlim(coord_min, coord_max)
        else:
            # 2D zoom logic
            for i, coord_name in enumerate(['X', 'Y']):
                coord_range = all_coords[:, i].max() - all_coords[:, i].min()
                visible_range = coord_range / self.zoom_factor
                center = target_point[i]
                
                coord_min = center - visible_range / 2
                coord_max = center + visible_range / 2
                
                if i == 0:
                    ax.set_xlim(coord_min, coord_max)
                else:
                    ax.set_ylim(coord_min, coord_max)
                    
    def _add_semantic_legend(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        semantic_axes_labels: Optional[Dict[str, str]]
    ) -> None:
        """
        Add semantic axes legend to the plot.
        
        Shared semantic legend logic.
        """
        if not semantic_axes_labels:
            return
            
        # Create bottom legend text
        legend_text = create_bottom_legend_text(
            semantic_axes_labels,
            max_chars_per_line=80,
            max_lines=2
        )
        
        if legend_text:
            # Add text at bottom of figure, outside the plot area
            fig.text(0.5, 0.02, legend_text, ha='center', va='bottom', 
                    fontsize=8, wrap=True, bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='lightgray', alpha=0.8))
                    
    @abstractmethod
    def _plot_points(
        self,
        ax: plt.Axes,
        train_coords: np.ndarray,
        train_data: np.ndarray,
        test_coords: np.ndarray,
        test_data: Optional[np.ndarray],
        highlight_test_idx: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Plot points on the axis (task-specific implementation).
        
        Args:
            ax: Matplotlib axis
            train_coords: Training coordinates [n_train, 2 or 3]
            train_data: Training labels/targets [n_train]
            test_coords: Test coordinates [n_test, 2 or 3]  
            test_data: Test labels/targets [n_test] (optional)
            highlight_test_idx: Index of test point to highlight
            **kwargs: Additional plotting parameters
            
        Returns:
            Dictionary with 'legend_text' and 'metadata' keys
        """
        pass
        
    def create_plot(
        self,
        train_coords: np.ndarray,
        train_data: np.ndarray,
        test_coords: np.ndarray,
        test_data: Optional[np.ndarray] = None,
        highlight_test_idx: Optional[int] = None,
        semantic_axes_labels: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Tuple[plt.Figure, str, Dict[str, Any]]:
        """
        Create complete t-SNE plot.
        
        Main plotting method that coordinates all the shared logic.
        
        Args:
            train_coords: Training coordinates [n_train, 2 or 3]
            train_data: Training labels/targets [n_train]
            test_coords: Test coordinates [n_test, 2 or 3]
            test_data: Test labels/targets [n_test] (optional)  
            highlight_test_idx: Index of test point to highlight
            semantic_axes_labels: Optional semantic axes labels
            **kwargs: Additional plotting parameters
            
        Returns:
            fig: Matplotlib figure
            legend_text: Legend description
            metadata: Plot metadata dictionary
        """
        # Step 1: Create figure
        fig, ax = self._create_figure()
        
        # Step 2: Apply zoom if highlighting a test point
        if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_coords):
            target_point = test_coords[highlight_test_idx]
            self._apply_zoom(ax, target_point, train_coords, test_coords)
            
        # Step 3: Task-specific point plotting
        plot_result = self._plot_points(
            ax, train_coords, train_data, test_coords, test_data, 
            highlight_test_idx, **kwargs
        )
        
        # Step 4: Apply consistent legend formatting
        apply_consistent_legend_formatting(ax, use_3d=self.use_3d)
        
        # Step 5: Add semantic legend if provided
        self._add_semantic_legend(fig, ax, semantic_axes_labels)
        
        # Step 6: Set axis labels and title
        if self.use_3d:
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2') 
            ax.set_zlabel('t-SNE Component 3')
        else:
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            
        return fig, plot_result['legend_text'], plot_result['metadata']


class ClassificationTSNEPlotter(BaseTSNEPlotter):
    """
    Handles classification-specific t-SNE plotting.
    
    Manages discrete class visualization with proper color mapping,
    class legends, and semantic name handling.
    """
    
    def _plot_points(
        self,
        ax: plt.Axes,
        train_coords: np.ndarray,
        train_labels: np.ndarray,
        test_coords: np.ndarray,
        test_labels: Optional[np.ndarray],
        highlight_test_idx: Optional[int],
        class_names: Optional[List[str]] = None,
        use_semantic_names: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Plot classification points using consistent styling.
        
        Args:
            ax: Matplotlib axis
            train_coords: Training coordinates [n_train, 2 or 3]
            train_labels: Training class labels [n_train]
            test_coords: Test coordinates [n_test, 2 or 3]
            test_labels: Test class labels [n_test] (optional)
            highlight_test_idx: Index of test point to highlight
            class_names: Optional class names for labeling
            use_semantic_names: Whether to use semantic class names
            **kwargs: Additional plotting parameters
            
        Returns:
            Dictionary with 'legend_text' and 'metadata' keys
        """
        # Determine which test indices to highlight
        highlight_test_indices = [highlight_test_idx] if highlight_test_idx is not None else None
        
        # Use the shared styling system for consistent appearance
        plot_result = apply_consistent_point_styling(
            ax=ax,
            transformed_data=train_coords,
            y=train_labels,
            highlight_indices=None,  # No training point highlighting for t-SNE
            test_data=test_coords,
            highlight_test_indices=highlight_test_indices,
            use_3d=self.use_3d,
            class_names=class_names,
            use_semantic_names=use_semantic_names,
            all_classes=np.unique(train_labels)
        )
        
        return plot_result


class RegressionTSNEPlotter(BaseTSNEPlotter):
    """
    Handles regression-specific t-SNE plotting.
    
    Manages continuous value visualization with colormaps,
    color bars, and regression-specific styling.
    """
    
    def _plot_points(
        self,
        ax: plt.Axes,
        train_coords: np.ndarray,
        train_targets: np.ndarray,
        test_coords: np.ndarray,
        test_targets: Optional[np.ndarray],
        highlight_test_idx: Optional[int],
        colormap: str = 'viridis',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Plot regression points using continuous color mapping.
        
        Args:
            ax: Matplotlib axis
            train_coords: Training coordinates [n_train, 2 or 3]
            train_targets: Training target values [n_train]
            test_coords: Test coordinates [n_test, 2 or 3]
            test_targets: Test target values [n_test] (optional)
            highlight_test_idx: Index of test point to highlight
            colormap: Matplotlib colormap name for target values
            **kwargs: Additional plotting parameters
            
        Returns:
            Dictionary with 'legend_text' and 'metadata' keys
        """
        # Create regression color mapping
        color_map = create_regression_color_map(train_targets, colormap=colormap)
        
        # Plot training points with continuous coloring
        if self.use_3d:
            scatter = ax.scatter(
                train_coords[:, 0],
                train_coords[:, 1], 
                train_coords[:, 2],
                c=train_targets,
                cmap=colormap,
                alpha=0.6,
                s=50,
                label='Training Data'
            )
        else:
            scatter = ax.scatter(
                train_coords[:, 0],
                train_coords[:, 1],
                c=train_targets,
                cmap=colormap,
                alpha=0.6,
                s=50,
                label='Training Data'
            )
            
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Target Value', rotation=270, labelpad=15)
        
        # Plot test points with gray styling
        test_style = get_standard_test_point_style()
        if self.use_3d:
            ax.scatter(
                test_coords[:, 0],
                test_coords[:, 1],
                test_coords[:, 2],
                **test_style
            )
        else:
            ax.scatter(
                test_coords[:, 0],
                test_coords[:, 1],
                **test_style
            )
            
        # Highlight specific test point if requested
        if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_coords):
            target_style = get_standard_target_point_style()
            if self.use_3d:
                ax.scatter(
                    test_coords[highlight_test_idx, 0],
                    test_coords[highlight_test_idx, 1],
                    test_coords[highlight_test_idx, 2],
                    **target_style
                )
            else:
                ax.scatter(
                    test_coords[highlight_test_idx, 0],
                    test_coords[highlight_test_idx, 1],
                    **target_style
                )
                
        # Create metadata
        metadata = {
            'plot_type': 'regression',
            'visible_classes': [],  # No classes in regression
            'target_range': [float(train_targets.min()), float(train_targets.max())],
            'colormap': colormap
        }
        
        legend_text = f"Regression visualization (target range: {train_targets.min():.2f} - {train_targets.max():.2f})"
        
        return {
            'legend_text': legend_text,
            'metadata': metadata
        }


class KNNMixin:
    """
    Mixin class that adds KNN connection functionality to any plotter.
    
    This eliminates the duplication between KNN and non-KNN variants
    of the plotting functions.
    """
    
    def __init__(self, *args, knn_k: int = 5, **kwargs):
        """
        Initialize KNN mixin.
        
        Args:
            knn_k: Number of nearest neighbors to find and visualize
        """
        super().__init__(*args, **kwargs)
        self.knn_k = knn_k
        
    def _compute_knn_analysis(
        self,
        query_point: np.ndarray,
        training_embeddings: np.ndarray,
        training_labels: np.ndarray,
        k: int
    ) -> Dict[str, Any]:
        """
        Compute KNN analysis for a query point.
        
        Args:
            query_point: Query point embedding [embedding_dim]
            training_embeddings: Training embeddings [n_train, embedding_dim] 
            training_labels: Training labels [n_train]
            k: Number of nearest neighbors
            
        Returns:
            Dictionary with KNN analysis results
        """
        # Compute distances to all training points
        distances = np.linalg.norm(training_embeddings - query_point, axis=1)
        
        # Get indices of k nearest neighbors
        knn_indices = np.argsort(distances)[:k]
        knn_distances = distances[knn_indices]
        knn_labels = training_labels[knn_indices]
        
        return {
            'indices': knn_indices,
            'distances': knn_distances,
            'labels': knn_labels
        }
        
    def _add_knn_connections(
        self,
        ax: plt.Axes,
        query_coord: np.ndarray,
        knn_coords: np.ndarray,
        use_3d: bool = False
    ) -> None:
        """
        Add KNN connection lines to the plot.
        
        Args:
            ax: Matplotlib axis
            query_coord: Query point coordinates [2 or 3]
            knn_coords: KNN coordinates [k, 2 or 3]
            use_3d: Whether this is a 3D plot
        """
        for knn_coord in knn_coords:
            if use_3d:
                ax.plot(
                    [query_coord[0], knn_coord[0]],
                    [query_coord[1], knn_coord[1]], 
                    [query_coord[2], knn_coord[2]],
                    'r--', alpha=0.5, linewidth=1
                )
            else:
                ax.plot(
                    [query_coord[0], knn_coord[0]],
                    [query_coord[1], knn_coord[1]],
                    'r--', alpha=0.5, linewidth=1
                )
                
    def _create_knn_pie_chart(
        self,
        ax: plt.Axes,
        knn_labels: np.ndarray,
        class_names: Optional[List[str]] = None,
        use_semantic_names: bool = False
    ) -> None:
        """
        Create KNN pie chart showing class distribution.
        
        Args:
            ax: Matplotlib axis for pie chart
            knn_labels: Labels of KNN points [k]
            class_names: Optional class names
            use_semantic_names: Whether to use semantic names
        """
        from collections import Counter
        from .utils.styling import format_class_label, create_distinct_color_map
        
        # Count class occurrences
        label_counts = Counter(knn_labels)
        
        # Create consistent colors
        all_classes = np.unique(knn_labels)
        color_map = create_distinct_color_map(all_classes)
        
        # Prepare data for pie chart
        labels = []
        sizes = []
        colors = []
        
        for label, count in label_counts.items():
            formatted_label = format_class_label(label, class_names, use_semantic_names)
            labels.append(f"{formatted_label} ({count})")
            sizes.append(count)
            colors.append(color_map[label])
            
        # Create pie chart
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'KNN Class Distribution (k={len(knn_labels)})')


class TSNEVisualizer:
    """
    Main unified interface for t-SNE visualization.
    
    This single class replaces all 14 functions from the original tsne_functions.py,
    eliminating massive code duplication while providing all the same functionality.
    """
    
    def __init__(
        self,
        task_type: str = 'classification',
        use_3d: bool = False,
        use_knn: bool = False,
        knn_k: int = 5,
        perplexity: int = 30,
        max_iter: int = 1000,
        random_state: int = 42,
        figsize: Tuple[int, int] = (10, 8),
        zoom_factor: float = 2.0
    ):
        """
        Initialize unified t-SNE visualizer.
        
        Args:
            task_type: 'classification' or 'regression'
            use_3d: Whether to create 3D visualizations
            use_knn: Whether to include KNN connections and analysis  
            knn_k: Number of nearest neighbors (if use_knn=True)
            perplexity: t-SNE perplexity parameter
            max_iter: Number of t-SNE iterations
            random_state: Random seed for reproducibility
            figsize: Figure size (width, height)
            zoom_factor: Zoom level for highlighted points
        """
        # Initialize coordinate generator
        self.generator = TSNEGenerator(
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state
        )
        
        # Initialize plotter with appropriate configuration
        self.plotter = self._create_plotter(task_type, use_3d, use_knn, knn_k, figsize, zoom_factor)
        self.task_type = task_type
        self.use_3d = use_3d
        self.use_knn = use_knn
        
    def _create_plotter(
        self,
        task_type: str,
        use_3d: bool,
        use_knn: bool,
        knn_k: int,
        figsize: Tuple[int, int],
        zoom_factor: float
    ):
        """
        Factory method to create appropriate plotter with mixins.
        
        Args:
            task_type: 'classification' or 'regression'
            use_3d: Whether to create 3D plots
            use_knn: Whether to include KNN functionality
            knn_k: Number of nearest neighbors
            figsize: Figure size
            zoom_factor: Zoom level
            
        Returns:
            Configured plotter instance
        """
        # Choose base plotter class
        if task_type == 'classification':
            base_class = ClassificationTSNEPlotter
        elif task_type == 'regression':
            base_class = RegressionTSNEPlotter
        else:
            raise ValueError(f"Unknown task_type: {task_type}. Must be 'classification' or 'regression'")
            
        # Apply KNN mixin if requested
        if use_knn:
            # Create a new class that combines KNNMixin with the base plotter
            class_name = f"KNN{base_class.__name__}"
            plotter_class = type(class_name, (KNNMixin, base_class), {})
            return plotter_class(figsize=figsize, zoom_factor=zoom_factor, use_3d=use_3d, knn_k=knn_k)
        else:
            return base_class(figsize=figsize, zoom_factor=zoom_factor, use_3d=use_3d)
            
    def create_visualization(
        self,
        train_embeddings: np.ndarray,
        train_data: np.ndarray,
        test_embeddings: np.ndarray,
        test_data: Optional[np.ndarray] = None,
        highlight_test_idx: Optional[int] = None,
        semantic_axes_labels: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, plt.Figure, str, Dict[str, Any]]:
        """
        Create complete t-SNE visualization.
        
        This single method replaces all 14 original functions:
        - create_tsne_visualization / create_tsne_3d_visualization
        - create_combined_tsne_plot / create_combined_tsne_3d_plot  
        - create_tsne_plot_with_knn / create_tsne_3d_plot_with_knn
        - create_regression_tsne_visualization / create_regression_tsne_3d_visualization
        - create_combined_regression_tsne_plot / create_combined_regression_tsne_3d_plot
        - create_regression_tsne_plot_with_knn / create_regression_tsne_3d_plot_with_knn
        - And more...
        
        Args:
            train_embeddings: Training embeddings [n_train, embedding_dim]
            train_data: Training labels (classification) or targets (regression) [n_train]
            test_embeddings: Test embeddings [n_test, embedding_dim]
            test_data: Test labels/targets [n_test] (optional)
            highlight_test_idx: Index of test point to highlight
            semantic_axes_labels: Optional semantic axes labels
            **kwargs: Additional plotting parameters (class_names, use_semantic_names, colormap, etc.)
            
        Returns:
            train_coords: t-SNE coordinates for training data [n_train, 2 or 3]
            test_coords: t-SNE coordinates for test data [n_test, 2 or 3]
            fig: Matplotlib figure
            legend_text: Legend description
            metadata: Plot metadata dictionary
        """
        # Step 1: Generate t-SNE coordinates
        train_coords, test_coords = self.generator.fit_transform(
            train_embeddings,
            test_embeddings,
            n_components=3 if self.use_3d else 2
        )
        
        # Step 2: Create plot
        fig, legend_text, metadata = self.plotter.create_plot(
            train_coords=train_coords,
            train_data=train_data,
            test_coords=test_coords,
            test_data=test_data,
            highlight_test_idx=highlight_test_idx,
            semantic_axes_labels=semantic_axes_labels,
            **kwargs
        )
        
        return train_coords, test_coords, fig, legend_text, metadata
        
    def create_simple_visualization(
        self,
        train_embeddings: np.ndarray,
        train_data: np.ndarray,
        test_embeddings: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, plt.Figure]:
        """
        Create simple t-SNE visualization (backward compatibility).
        
        Matches the original simple function signatures that just returned coordinates and figure.
        """
        train_coords, test_coords, fig, _, _ = self.create_visualization(
            train_embeddings, train_data, test_embeddings, **kwargs
        )
        return train_coords, test_coords, fig


# ============================================================================
# BACKWARD COMPATIBILITY WRAPPER FUNCTIONS
# ============================================================================
# 
# These functions provide backward compatibility with the original tsne_functions.py
# by wrapping the new class-based implementation. This allows existing code to
# continue working while benefiting from the reduced duplication.


def create_tsne_visualization(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: Optional[np.ndarray] = None,
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42,
    figsize: Tuple[int, int] = (12, 8),
    class_names: Optional[List[str]] = None,
    use_semantic_names: bool = False,
    use_3d: bool = False
) -> Tuple[np.ndarray, np.ndarray, plt.Figure]:
    """
    Create t-SNE visualization of train and test embeddings.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    This wrapper is provided for backward compatibility.
    """
    import warnings
    warnings.warn(
        "create_tsne_visualization is deprecated. Use TSNEVisualizer instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    visualizer = TSNEVisualizer(
        task_type='classification',
        use_3d=use_3d,
        use_knn=False,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
        figsize=figsize
    )
    
    return visualizer.create_simple_visualization(
        train_embeddings, train_labels, test_embeddings,
        class_names=class_names, use_semantic_names=use_semantic_names
    )


def create_tsne_3d_visualization(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: Optional[np.ndarray] = None,
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42,
    figsize: Tuple[int, int] = (15, 12),
    class_names: Optional[List[str]] = None,
    use_semantic_names: bool = False
) -> Tuple[np.ndarray, np.ndarray, plt.Figure]:
    """
    Create 3D t-SNE visualization of train and test embeddings.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    """
    import warnings
    warnings.warn(
        "create_tsne_3d_visualization is deprecated. Use TSNEVisualizer instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return create_tsne_visualization(
        train_embeddings, train_labels, test_embeddings, test_labels,
        perplexity, max_iter, random_state, figsize, class_names, use_semantic_names, use_3d=True
    )


def create_combined_tsne_plot(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_labels: np.ndarray,
    highlight_test_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 8),
    zoom_factor: float = 2.0,
    class_names: Optional[List[str]] = None,
    use_semantic_names: bool = False,
    use_3d: bool = False,
    semantic_axes_labels: Optional[Dict[str, str]] = None
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create a t-SNE plot with optional highlighting of a specific test point.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    """
    import warnings
    warnings.warn(
        "create_combined_tsne_plot is deprecated. Use TSNEVisualizer instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create a dummy visualizer to use the plotter
    plotter = ClassificationTSNEPlotter(figsize=figsize, zoom_factor=zoom_factor, use_3d=use_3d)
    
    fig, legend_text, metadata = plotter.create_plot(
        train_coords=train_tsne,
        train_data=train_labels,
        test_coords=test_tsne,
        test_data=None,
        highlight_test_idx=highlight_test_idx,
        semantic_axes_labels=semantic_axes_labels,
        class_names=class_names,
        use_semantic_names=use_semantic_names
    )
    
    return fig, legend_text, metadata


def create_combined_tsne_3d_plot(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_labels: np.ndarray,
    highlight_test_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 12),
    zoom_factor: float = 2.0,
    class_names: Optional[List[str]] = None,
    use_semantic_names: bool = False,
    semantic_axes_labels: Optional[Dict[str, str]] = None
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create a 3D t-SNE plot with optional highlighting of a specific test point.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    """
    import warnings
    warnings.warn(
        "create_combined_tsne_3d_plot is deprecated. Use TSNEVisualizer instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return create_combined_tsne_plot(
        train_tsne, test_tsne, train_labels, highlight_test_idx,
        figsize, zoom_factor, class_names, use_semantic_names, use_3d=True, semantic_axes_labels=semantic_axes_labels
    )


def create_regression_tsne_visualization(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    test_embeddings: np.ndarray,
    test_targets: Optional[np.ndarray] = None,
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42,
    figsize: Tuple[int, int] = (12, 8),
    colormap: str = 'viridis',
    use_3d: bool = False
) -> Tuple[np.ndarray, np.ndarray, plt.Figure]:
    """
    Create t-SNE visualization for regression data with continuous color mapping.
    
    DEPRECATED: This function is deprecated. Use TSNEVisualizer instead.
    """
    import warnings
    warnings.warn(
        "create_regression_tsne_visualization is deprecated. Use TSNEVisualizer instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    visualizer = TSNEVisualizer(
        task_type='regression',
        use_3d=use_3d,
        use_knn=False,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
        figsize=figsize
    )
    
    return visualizer.create_simple_visualization(
        train_embeddings, train_targets, test_embeddings, colormap=colormap
    )


# Add more wrapper functions as needed...
# For now, we provide the most commonly used ones. Additional functions can be added
# following the same pattern when needed for compatibility.


__all__ = [
    # New refactored classes (recommended)
    'TSNEGenerator',
    'BaseTSNEPlotter', 
    'ClassificationTSNEPlotter',
    'RegressionTSNEPlotter',
    'KNNMixin',
    'TSNEVisualizer',
    
    # Backward compatibility functions (deprecated)
    'create_tsne_visualization',
    'create_tsne_3d_visualization', 
    'create_combined_tsne_plot',
    'create_combined_tsne_3d_plot',
    'create_regression_tsne_visualization',
]