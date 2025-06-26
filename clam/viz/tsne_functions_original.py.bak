"""
T-SNE visualization utilities for tabular data embeddings.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
from sklearn.manifold import TSNE
from typing import Tuple, Optional, List, Dict, Union
import io
import base64
from PIL import Image
import warnings

# Import shared styling utilities
from .utils.styling import (
    get_distinct_colors,
    create_distinct_color_map,
    get_class_color_name_map,
    create_class_legend,
    format_class_label,
    create_regression_color_map,
    apply_consistent_point_styling,
    apply_consistent_legend_formatting,
    get_standard_test_point_style,
    get_standard_target_point_style,
    get_standard_training_point_style
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

__all__ = [
    'create_tsne_visualization',
    'create_tsne_3d_visualization',
    'create_class_legend',
    'create_combined_tsne_plot',
    'create_combined_tsne_3d_plot',
    'create_tsne_plot_with_knn',
    'create_tsne_3d_plot_with_knn',
    'create_regression_tsne_visualization',
    'create_regression_tsne_3d_visualization',
    'create_combined_regression_tsne_plot',
    'create_combined_regression_tsne_3d_plot',
    'create_regression_tsne_plot_with_knn',
    'create_regression_tsne_3d_plot_with_knn',
    'get_distinct_colors',
    'create_distinct_color_map',
    'get_class_color_name_map',
    'create_regression_color_map'
]

logger = logging.getLogger(__name__)


def add_semantic_legend_to_plot(fig, ax, semantic_axes_labels, max_chars_per_line=80, max_lines=2):
    """
    Add a bottom legend for semantic axes to prevent overlap with plot elements.
    
    Args:
        fig: Matplotlib figure object
        ax: Matplotlib axes object
        semantic_axes_labels: Dictionary mapping axis names to semantic descriptions
        max_chars_per_line: Maximum characters per line in legend
        max_lines: Maximum number of lines in legend
    """
    if not semantic_axes_labels:
        return
    
    # Create bottom legend text
    legend_text = create_bottom_legend_text(
        semantic_axes_labels,
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines
    )
    
    if legend_text:
        # Add text at bottom of figure, outside the plot area
        fig.text(0.5, 0.02, legend_text, ha='center', va='bottom', 
                fontsize=8, wrap=True, bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='lightgray', alpha=0.8))


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
    
    Args:
        train_embeddings: Training embeddings [n_train, embedding_dim]
        train_labels: Training labels [n_train]
        test_embeddings: Test embeddings [n_test, embedding_dim]
        test_labels: Optional test labels for validation [n_test]
        perplexity: t-SNE perplexity parameter
        max_iter: Number of t-SNE iterations
        random_state: Random seed for reproducibility
        figsize: Figure size (width, height)
        class_names: Optional class names for labeling
        use_semantic_names: Whether to use semantic class names
        use_3d: Whether to create 3D visualization (default: False for 2D)
        
    Returns:
        train_tsne: t-SNE coordinates for training data [n_train, 2 or 3]
        test_tsne: t-SNE coordinates for test data [n_test, 2 or 3]
        fig: Matplotlib figure object (2D or 3D depending on use_3d)
    """
    logger.info(f"Creating t-SNE visualization with {len(train_embeddings)} train and {len(test_embeddings)} test samples")
    
    # Combine embeddings for joint t-SNE
    combined_embeddings = np.vstack([train_embeddings, test_embeddings])
    n_train = len(train_embeddings)
    
    # Adjust perplexity if needed based on data size
    effective_perplexity = min(perplexity, (len(combined_embeddings) - 1) // 3)
    if effective_perplexity != perplexity:
        logger.warning(f"Adjusting perplexity from {perplexity} to {effective_perplexity} due to small dataset size")
    
    # Apply t-SNE
    logger.info(f"Running t-SNE with perplexity={effective_perplexity}, max_iter={max_iter}")
    tsne = TSNE(
        n_components=3 if use_3d else 2,
        perplexity=effective_perplexity,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1
    )
    
    # Suppress numerical warnings during t-SNE computation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
        tsne_results = tsne.fit_transform(combined_embeddings)
    
    # Split back into train and test
    train_tsne = tsne_results[:n_train]
    test_tsne = tsne_results[n_train:]
    
    # Create visualization - handle both 2D and 3D
    unique_classes = np.unique(train_labels)
    class_color_map = create_distinct_color_map(unique_classes)
    
    if use_3d:
        # Create 3D visualization with four different views
        fig = plt.figure(figsize=(15, 12) if figsize == (12, 8) else figsize)
        
        # Create four subplots for different 3D views
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # View configurations: (elevation, azimuth, title)
        views = [
            (20, -60, "Isometric View"),  # Default 3D view
            (0, -90, "Front View (XZ)"),   # Front view
            (0, 0, "Side View (YZ)"),      # Side view  
            (90, 0, "Top View (XY)")       # Top view
        ]
        
        for i, (elev, azim, title) in enumerate(views):
            ax = fig.add_subplot(gs[i//2, i%2], projection='3d')
            
            # Plot training points with colors
            for class_label in unique_classes:
                mask = train_labels == class_label
                ax.scatter(
                    train_tsne[mask, 0], train_tsne[mask, 1], train_tsne[mask, 2],
                    c=[class_color_map[class_label]], 
                    label=format_class_label(class_label, class_names, use_semantic_names),
                    alpha=0.7,
                    s=50
                )
            
            # Plot test points in gray
            ax.scatter(
                test_tsne[:, 0], test_tsne[:, 1], test_tsne[:, 2],
                c='lightgray',
                label='Test Points' if i == 0 else "",  # Only label once
                alpha=0.8,
                s=60,
                marker='s'  # Square markers for test points
            )
            
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2') 
            ax.set_zlabel('t-SNE Dimension 3')
            ax.set_title(title)
            ax.view_init(elev=elev, azim=azim)
            
            # Only show legend on first subplot
            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle('3D t-SNE Visualization: Training (Colored) vs Test (Gray) Data', fontsize=14)
    else:
        # Create 2D visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot training points with colors
        for class_label in unique_classes:
            mask = train_labels == class_label
            ax.scatter(
                train_tsne[mask, 0], train_tsne[mask, 1],
                c=[class_color_map[class_label]], 
                label=format_class_label(class_label, class_names, use_semantic_names),
                alpha=0.7,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
        
        # Plot test points in gray
        ax.scatter(
            test_tsne[:, 0], test_tsne[:, 1],
            c='lightgray',
            label='Test Points (Light Gray)',
            alpha=0.8,
            s=60,
            edgecolors='black',
            linewidth=0.8,
            marker='s'  # Square markers for test points
        )
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('t-SNE Visualization: Training (Colored) vs Test (Gray) Data')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    logger.info(f"{'3D' if use_3d else '2D'} t-SNE visualization created successfully")
    return train_tsne, test_tsne, fig


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
    
    Args:
        train_embeddings: Training embeddings [n_train, embedding_dim]
        train_labels: Training labels [n_train]
        test_embeddings: Test embeddings [n_test, embedding_dim]
        test_labels: Optional test labels for validation [n_test]
        perplexity: t-SNE perplexity parameter
        max_iter: Number of t-SNE iterations
        random_state: Random seed for reproducibility
        figsize: Figure size (width, height)
        
    Returns:
        train_tsne: 3D t-SNE coordinates for training data [n_train, 3]
        test_tsne: 3D t-SNE coordinates for test data [n_test, 3]
        fig: Matplotlib figure object with 3D subplot
    """
    logger.info(f"Creating 3D t-SNE visualization with {len(train_embeddings)} train and {len(test_embeddings)} test samples")
    
    # Combine embeddings for joint t-SNE
    combined_embeddings = np.vstack([train_embeddings, test_embeddings])
    n_train = len(train_embeddings)
    
    # Adjust perplexity if needed based on data size
    effective_perplexity = min(perplexity, (len(combined_embeddings) - 1) // 3)
    if effective_perplexity != perplexity:
        logger.warning(f"Adjusting perplexity from {perplexity} to {effective_perplexity} due to small dataset size")
    
    # Apply 3D t-SNE
    logger.info(f"Running 3D t-SNE with perplexity={effective_perplexity}, max_iter={max_iter}")
    tsne = TSNE(
        n_components=3,  # 3D instead of 2D
        perplexity=effective_perplexity,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1
    )
    
    # Suppress numerical warnings during t-SNE computation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
        tsne_results = tsne.fit_transform(combined_embeddings)
    
    # Split back into train and test
    train_tsne = tsne_results[:n_train]
    test_tsne = tsne_results[n_train:]
    
    # Create 3D visualization
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique classes and colors
    unique_classes = np.unique(train_labels)
    class_color_map = create_distinct_color_map(unique_classes)
    class_color_names = get_class_color_name_map(unique_classes)
    
    # Plot training points with colors
    for i, class_label in enumerate(unique_classes):
        mask = train_labels == class_label
        ax.scatter(
            train_tsne[mask, 0], train_tsne[mask, 1], train_tsne[mask, 2],
            c=[class_color_map[class_label]], 
            label=format_class_label(class_label, class_names, use_semantic_names),
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Plot test points in gray
    ax.scatter(
        test_tsne[:, 0], test_tsne[:, 1], test_tsne[:, 2],
        c='lightgray',
        label='Test Points (Light Gray)',
        alpha=0.8,
        s=60,
        edgecolors='black',
        linewidth=0.8,
        marker='s'  # Square markers for test points
    )
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    ax.set_title('3D t-SNE Visualization: Training (Colored) vs Test (Gray) Data')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    logger.info("3D t-SNE visualization created successfully")
    return train_tsne, test_tsne, fig









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
    
    Args:
        train_tsne: t-SNE coordinates for training data [n_train, 2 or 3]
        test_tsne: t-SNE coordinates for test data [n_test, 2 or 3]
        train_labels: Training labels [n_train]
        highlight_test_idx: Index of test point to highlight (optional)
        figsize: Figure size (width, height)
        zoom_factor: Zoom level (2.0 = 200% zoom, showing 50% of the range)
        class_names: Optional class names for labeling
        use_semantic_names: Whether to use semantic class names
        use_3d: Whether to create 3D visualization (default: False for 2D)
        
    Returns:
        fig: Matplotlib figure object (2D or 3D depending on use_3d)
        legend_text: Text description of the legend
        metadata: Dictionary with plot metadata
    """
    # Create figure and axis - handle both 2D and 3D
    if use_3d:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique classes and create consistent color mapping
    unique_classes = np.unique(train_labels)
    class_color_map = create_distinct_color_map(unique_classes)
    class_color_names = get_class_color_name_map(unique_classes)
    
    # Apply zoom if highlighting a test point
    if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_tsne) and zoom_factor > 1.0:
        # Get the target point coordinates
        target_point = test_tsne[highlight_test_idx]
        
        # Calculate the visible range based on zoom factor
        # zoom_factor = 2.0 means we show 1/2 of the original range
        visible_fraction = 1.0 / zoom_factor
        
        # Get original data ranges
        all_points = np.vstack([train_tsne, test_tsne])
        x_range = np.ptp(all_points[:, 0])  # peak-to-peak (max - min)
        y_range = np.ptp(all_points[:, 1])
        
        # Calculate zoom window
        x_window = x_range * visible_fraction
        y_window = y_range * visible_fraction
        
        # Set axis limits centered on target point
        x_min = target_point[0] - x_window / 2
        x_max = target_point[0] + x_window / 2
        y_min = target_point[1] - y_window / 2
        y_max = target_point[1] + y_window / 2
        
        # For 3D, also handle z-axis
        if use_3d and train_tsne.shape[1] >= 3:
            z_range = np.ptp(all_points[:, 2])
            z_window = z_range * visible_fraction
            z_min = target_point[2] - z_window / 2
            z_max = target_point[2] + z_window / 2
            
            # Filter points to only show those within the zoom window (3D)
            train_mask = ((train_tsne[:, 0] >= x_min) & (train_tsne[:, 0] <= x_max) & 
                         (train_tsne[:, 1] >= y_min) & (train_tsne[:, 1] <= y_max) &
                         (train_tsne[:, 2] >= z_min) & (train_tsne[:, 2] <= z_max))
            test_mask = ((test_tsne[:, 0] >= x_min) & (test_tsne[:, 0] <= x_max) & 
                        (test_tsne[:, 1] >= y_min) & (test_tsne[:, 1] <= y_max) &
                        (test_tsne[:, 2] >= z_min) & (test_tsne[:, 2] <= z_max))
        else:
            # Filter points to only show those within the zoom window (2D)
            train_mask = ((train_tsne[:, 0] >= x_min) & (train_tsne[:, 0] <= x_max) & 
                         (train_tsne[:, 1] >= y_min) & (train_tsne[:, 1] <= y_max))
            test_mask = ((test_tsne[:, 0] >= x_min) & (test_tsne[:, 0] <= x_max) & 
                        (test_tsne[:, 1] >= y_min) & (test_tsne[:, 1] <= y_max))
        
        train_tsne_visible = train_tsne[train_mask]
        train_labels_visible = train_labels[train_mask]
        test_tsne_visible = test_tsne[test_mask]
        
        # Set axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if use_3d and hasattr(ax, 'set_zlim'):
            ax.set_zlim(z_min, z_max)
    else:
        # No zoom, show all points
        train_tsne_visible = train_tsne
        train_labels_visible = train_labels
        test_tsne_visible = test_tsne
    
    # Plot training points with colors (only visible ones)
    for class_label in unique_classes:
        mask = train_labels_visible == class_label
        if np.any(mask):  # Only plot if there are points of this class in view
            if use_3d and train_tsne_visible.shape[1] >= 3:
                ax.scatter(
                    train_tsne_visible[mask, 0], train_tsne_visible[mask, 1], train_tsne_visible[mask, 2],
                    c=[class_color_map[class_label]], 
                    label=format_class_label(class_label, class_names, use_semantic_names, "Training Class"),
                    alpha=0.7,
                    s=50
                )
            else:
                ax.scatter(
                    train_tsne_visible[mask, 0], train_tsne_visible[mask, 1],
                    c=[class_color_map[class_label]], 
                    label=format_class_label(class_label, class_names, use_semantic_names, "Training Class"),
                    alpha=0.7,
                    s=50,
                    edgecolors='black',
                    linewidth=0.5
                )
    
    # Plot all test points in gray (only visible ones)
    if len(test_tsne_visible) > 0:
        if use_3d and test_tsne_visible.shape[1] >= 3:
            ax.scatter(
                test_tsne_visible[:, 0], test_tsne_visible[:, 1], test_tsne_visible[:, 2],
                c='lightgray',
                alpha=0.6,
                s=60,
                marker='s',
                label='Test Points (Light Gray)'
            )
        else:
            ax.scatter(
                test_tsne_visible[:, 0], test_tsne_visible[:, 1],
                c='lightgray',
                alpha=0.6,
                s=60,
                edgecolors='gray',
                linewidth=0.8,
                marker='s',
                label='Test Points (Light Gray)'
            )
    
    # Highlight specific test point if requested
    if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_tsne):
        if use_3d and test_tsne.shape[1] >= 3:
            ax.scatter(
                test_tsne[highlight_test_idx, 0], test_tsne[highlight_test_idx, 1], test_tsne[highlight_test_idx, 2],
                c='red',
                s=120,
                marker='*',
                label='Query Point (Red Star)'
            )
        else:
            ax.scatter(
                test_tsne[highlight_test_idx, 0], test_tsne[highlight_test_idx, 1],
                c='red',
                s=120,
                edgecolors='darkred',
                linewidth=2,
                marker='*',
                label='Query Point (Red Star)',
                zorder=10
            )
    
    # Set axis labels with compact semantic information to prevent overlap
    compact_labels = create_compact_axis_labels(semantic_axes_labels, max_chars_per_line=30)
    
    if compact_labels and 'X' in compact_labels:
        ax.set_xlabel(compact_labels['X'])
    else:
        ax.set_xlabel('t-SNE Dimension 1')
    
    if compact_labels and 'Y' in compact_labels:
        ax.set_ylabel(compact_labels['Y'])
    else:
        ax.set_ylabel('t-SNE Dimension 2')
    
    if use_3d and hasattr(ax, 'set_zlabel'):
        if compact_labels and 'Z' in compact_labels:
            ax.set_zlabel(compact_labels['Z'])
        else:
            ax.set_zlabel('t-SNE Dimension 3')
    
    # Add bottom legend for full semantic information
    add_semantic_legend_to_plot(fig, ax, semantic_axes_labels)
    
    if highlight_test_idx is not None:
        if zoom_factor > 1.0:
            ax.set_title(f't-SNE Visualization - Query Point {highlight_test_idx} (Zoom: {zoom_factor:.1f}x)')
        else:
            ax.set_title(f't-SNE Visualization - Query Point {highlight_test_idx}')
    else:
        ax.set_title('t-SNE Visualization - Training vs Test Data')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create legend text
    legend_text = create_class_legend(unique_classes, class_color_map, class_names, use_semantic_names)
    
    # Create metadata with visible classes from legend
    from .utils.styling import extract_visible_classes_from_legend
    
    metadata = {
        'n_train_points': len(train_tsne),
        'n_test_points': len(test_tsne),
        'n_classes': len(unique_classes),
        'classes': unique_classes.tolist(),  # Legacy field
        'visible_classes': extract_visible_classes_from_legend(unique_classes, class_names, use_semantic_names),
        'highlighted_point': highlight_test_idx,
        'zoom_factor': zoom_factor if highlight_test_idx is not None else None,
        'n_visible_train': len(train_tsne_visible) if 'train_tsne_visible' in locals() else len(train_tsne),
        'n_visible_test': len(test_tsne_visible) if 'test_tsne_visible' in locals() else len(test_tsne),
        'plot_type': 'classification'
    }
    
    return fig, legend_text, metadata


def create_combined_tsne_3d_plot(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_labels: np.ndarray,
    highlight_test_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (20, 15),
    viewing_angles: Optional[List[Tuple[int, int]]] = None,
    zoom_factor: float = 2.0,
    class_names: Optional[List[str]] = None,
    use_semantic_names: bool = False,
    semantic_axes_labels: Optional[Dict[str, str]] = None
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create multiple 3D t-SNE plots with different viewing angles for comprehensive spatial understanding.
    
    Args:
        train_tsne: 3D t-SNE coordinates for training data [n_train, 3]
        test_tsne: 3D t-SNE coordinates for test data [n_test, 3]
        train_labels: Training labels [n_train]
        highlight_test_idx: Index of test point to highlight (optional)
        figsize: Figure size (width, height)
        viewing_angles: List of (elevation, azimuth) tuples for different views
        zoom_factor: Zoom level (2.0 = 200% zoom, showing 50% of the range)
        
    Returns:
        fig: Matplotlib figure object with multiple 3D subplots
        legend_text: Text description of the legend
        metadata: Dictionary with plot metadata
    """
    if viewing_angles is None:
        # Default viewing angles: front, side, top, and isometric views
        viewing_angles = [
            (20, 45),   # Isometric view
            (0, 0),     # Front view (XZ plane)
            (0, 90),    # Side view (YZ plane)
            (90, 0),    # Top view (XY plane)
        ]
    
    # Create subplots for multiple views
    n_views = len(viewing_angles)
    fig = plt.figure(figsize=figsize)
    
    # Get unique classes and create consistent color mapping
    unique_classes = np.unique(train_labels)
    class_color_map = create_distinct_color_map(unique_classes)
    class_color_names = get_class_color_name_map(unique_classes)
    
    # Apply zoom if highlighting a test point
    if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_tsne) and zoom_factor > 1.0:
        # Get the target point coordinates
        target_point = test_tsne[highlight_test_idx]
        
        # Calculate the visible range based on zoom factor
        visible_fraction = 1.0 / zoom_factor
        
        # Get original data ranges
        all_points = np.vstack([train_tsne, test_tsne])
        x_range = np.ptp(all_points[:, 0])
        y_range = np.ptp(all_points[:, 1])
        z_range = np.ptp(all_points[:, 2])
        
        # Calculate zoom window
        x_window = x_range * visible_fraction
        y_window = y_range * visible_fraction
        z_window = z_range * visible_fraction
        
        # Set axis limits centered on target point
        x_min = target_point[0] - x_window / 2
        x_max = target_point[0] + x_window / 2
        y_min = target_point[1] - y_window / 2
        y_max = target_point[1] + y_window / 2
        z_min = target_point[2] - z_window / 2
        z_max = target_point[2] + z_window / 2
        
        # Filter points to only show those within the zoom window
        train_mask = ((train_tsne[:, 0] >= x_min) & (train_tsne[:, 0] <= x_max) & 
                     (train_tsne[:, 1] >= y_min) & (train_tsne[:, 1] <= y_max) &
                     (train_tsne[:, 2] >= z_min) & (train_tsne[:, 2] <= z_max))
        test_mask = ((test_tsne[:, 0] >= x_min) & (test_tsne[:, 0] <= x_max) & 
                    (test_tsne[:, 1] >= y_min) & (test_tsne[:, 1] <= y_max) &
                    (test_tsne[:, 2] >= z_min) & (test_tsne[:, 2] <= z_max))
        
        train_tsne_visible = train_tsne[train_mask]
        train_labels_visible = train_labels[train_mask]
        test_tsne_visible = test_tsne[test_mask]
    else:
        # No zoom, show all points
        train_tsne_visible = train_tsne
        train_labels_visible = train_labels
        test_tsne_visible = test_tsne
        x_min = x_max = y_min = y_max = z_min = z_max = None
    
    view_names = ['Isometric View', 'Front View (XZ)', 'Side View (YZ)', 'Top View (XY)']
    
    for i, (elev, azim) in enumerate(viewing_angles):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        
        # Plot training points with colors (only visible ones)
        for class_label in unique_classes:
            mask = train_labels_visible == class_label
            if np.any(mask):  # Only plot if there are points of this class in view
                ax.scatter(
                    train_tsne_visible[mask, 0], train_tsne_visible[mask, 1], train_tsne_visible[mask, 2],
                    c=[class_color_map[class_label]], 
                    label=format_class_label(class_label, class_names, use_semantic_names, "Training Class") if i == 0 else "",  # Only show legend on first plot
                    alpha=0.7,
                    s=40,
                    edgecolors='black',
                    linewidth=0.5
                )
        
        # Plot all test points in gray (only visible ones)
        if len(test_tsne_visible) > 0:
            ax.scatter(
                test_tsne_visible[:, 0], test_tsne_visible[:, 1], test_tsne_visible[:, 2],
                c='lightgray',
                alpha=0.6,
                s=50,
                edgecolors='gray',
                linewidth=0.8,
                marker='s',
                label='Test Points (Light Gray)' if i == 0 else ""  # Only show legend on first plot
            )
        
        # Highlight specific test point if requested
        if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_tsne):
            ax.scatter(
                test_tsne[highlight_test_idx, 0], test_tsne[highlight_test_idx, 1], test_tsne[highlight_test_idx, 2],
                c='red',
                s=120,
                edgecolors='darkred',
                linewidth=2,
                marker='*',
                label='Query Point' if i == 0 else "",  # Only show legend on first plot
                zorder=10
            )
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Set axis limits if zoomed
        if x_min is not None:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
        
        # Labels and title with compact semantic information to prevent overlap
        compact_labels = create_compact_axis_labels(semantic_axes_labels, max_chars_per_line=25)
        
        if compact_labels and 'X' in compact_labels:
            ax.set_xlabel(compact_labels['X'])
        else:
            ax.set_xlabel('t-SNE Dim 1')
        
        if compact_labels and 'Y' in compact_labels:
            ax.set_ylabel(compact_labels['Y'])
        else:
            ax.set_ylabel('t-SNE Dim 2')
        
        if compact_labels and 'Z' in compact_labels:
            ax.set_zlabel(compact_labels['Z'])
        else:
            ax.set_zlabel('t-SNE Dim 3')
        
        # Add bottom legend for full semantic information
        add_semantic_legend_to_plot(fig, ax, semantic_axes_labels, max_chars_per_line=60)
        
        # Use shorter view names if available
        view_name = view_names[i] if i < len(view_names) else f'View {i+1}'
        if highlight_test_idx is not None:
            if zoom_factor > 1.0:
                ax.set_title(f'{view_name} - Query Point {highlight_test_idx} (Zoom: {zoom_factor:.1f}x)')
            else:
                ax.set_title(f'{view_name} - Query Point {highlight_test_idx}')
        else:
            ax.set_title(f'{view_name}')
        
        # Only add legend to the first subplot to avoid clutter
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Create legend text
    legend_text = create_class_legend(unique_classes, class_color_map, class_names, use_semantic_names)
    legend_text += f"\n\nThis visualization shows {n_views} different viewing angles of the same 3D t-SNE space:"
    for i, (elev, azim) in enumerate(viewing_angles):
        view_name = view_names[i] if i < len(view_names) else f'View {i+1}'
        legend_text += f"\n- {view_name}: Elevation={elev}°, Azimuth={azim}°"
    
    # Create metadata with visible classes from legend
    from .utils.styling import extract_visible_classes_from_legend
    
    metadata = {
        'n_train_points': len(train_tsne),
        'n_test_points': len(test_tsne),
        'n_classes': len(unique_classes),
        'classes': unique_classes.tolist(),
        'visible_classes': extract_visible_classes_from_legend(unique_classes, class_names, use_semantic_names),
        'highlighted_point': highlight_test_idx,
        'viewing_angles': viewing_angles,
        'n_views': n_views,
        'is_3d': True,
        'zoom_factor': zoom_factor if highlight_test_idx is not None else None,
        'n_visible_train': len(train_tsne_visible) if 'train_tsne_visible' in locals() else len(train_tsne),
        'n_visible_test': len(test_tsne_visible) if 'test_tsne_visible' in locals() else len(test_tsne)
    }
    
    return fig, legend_text, metadata


def create_tsne_plot_with_knn(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_labels: np.ndarray,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    highlight_test_idx: Optional[int] = None,
    k: int = 5,
    figsize: Tuple[int, int] = (14, 10),
    zoom_factor: float = 2.0,
    class_names: Optional[List[str]] = None,
    use_semantic_names: bool = False
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create a 2D t-SNE plot with KNN pie chart showing neighbor class distribution.
    
    Args:
        train_tsne: 2D t-SNE coordinates for training data [n_train, 2]
        test_tsne: 2D t-SNE coordinates for test data [n_test, 2]
        train_labels: Training labels [n_train]
        train_embeddings: Original training embeddings for KNN [n_train, embedding_dim]
        test_embeddings: Original test embeddings for KNN [n_test, embedding_dim]
        highlight_test_idx: Index of test point to highlight and show KNN pie chart
        k: Number of nearest neighbors to analyze
        figsize: Figure size (width, height)
        zoom_factor: Zoom level (2.0 = 200% zoom, showing 50% of the range)
        
    Returns:
        fig: Matplotlib figure object
        legend_text: Text description of the legend
        metadata: Dictionary with plot metadata
    """
    # Create figure with subplot for main plot and optional pie chart
    if highlight_test_idx is not None:
        fig = plt.figure(figsize=figsize)
        # Create a grid: main plot takes ~70% width, pie chart takes ~30% width
        gs = fig.add_gridspec(1, 5, width_ratios=[2.5, 2.5, 2, 1.5, 1.5], hspace=0.1, wspace=0.15)
        ax = fig.add_subplot(gs[0, :3])  # Main plot spans first 3 columns
        ax_pie = fig.add_subplot(gs[0, 3:])  # Pie chart spans last 2 columns (bigger)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_pie = None
    
    # Get unique classes and create consistent color mapping
    unique_classes = np.unique(train_labels)
    class_color_map = create_distinct_color_map(unique_classes)
    class_color_names = get_class_color_name_map(unique_classes)
    
    # Apply zoom if highlighting a test point
    if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_tsne) and zoom_factor > 1.0:
        # Get the target point coordinates
        target_point = test_tsne[highlight_test_idx]
        
        # Calculate the visible range based on zoom factor
        visible_fraction = 1.0 / zoom_factor
        
        # Get original data ranges
        all_points = np.vstack([train_tsne, test_tsne])
        x_range = np.ptp(all_points[:, 0])
        y_range = np.ptp(all_points[:, 1])
        
        # Calculate zoom window
        x_window = x_range * visible_fraction
        y_window = y_range * visible_fraction
        
        # Set axis limits centered on target point
        x_min = target_point[0] - x_window / 2
        x_max = target_point[0] + x_window / 2
        y_min = target_point[1] - y_window / 2
        y_max = target_point[1] + y_window / 2
        
        # Filter points to only show those within the zoom window
        train_mask = ((train_tsne[:, 0] >= x_min) & (train_tsne[:, 0] <= x_max) & 
                     (train_tsne[:, 1] >= y_min) & (train_tsne[:, 1] <= y_max))
        test_mask = ((test_tsne[:, 0] >= x_min) & (test_tsne[:, 0] <= x_max) & 
                    (test_tsne[:, 1] >= y_min) & (test_tsne[:, 1] <= y_max))
        
        train_tsne_visible = train_tsne[train_mask]
        train_labels_visible = train_labels[train_mask]
        test_tsne_visible = test_tsne[test_mask]
        
        # Set axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        # No zoom, show all points
        train_tsne_visible = train_tsne
        train_labels_visible = train_labels
        test_tsne_visible = test_tsne
    
    # Plot training points with colors (only visible ones)
    for class_label in unique_classes:
        mask = train_labels_visible == class_label
        if np.any(mask):
            ax.scatter(
                train_tsne_visible[mask, 0], train_tsne_visible[mask, 1],
                c=[class_color_map[class_label]], 
                label=format_class_label(class_label, class_names, use_semantic_names, "Training Class"),
                alpha=0.7,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
    
    # Plot all test points in gray (only visible ones)
    if len(test_tsne_visible) > 0:
        ax.scatter(
            test_tsne_visible[:, 0], test_tsne_visible[:, 1],
            c='lightgray',
            alpha=0.6,
            s=60,
            edgecolors='gray',
            linewidth=0.8,
            marker='s',
            label='Test Points (Light Gray)'
        )
    
    # If highlighting a specific test point, show KNN connections
    knn_info = {}
    if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_tsne):
        # Import KNN utilities
        from clam.models.knn_utils import find_knn_in_embedding_space
        
        # Find KNN for this specific test point
        query_embedding = test_embeddings[highlight_test_idx:highlight_test_idx+1]
        distances, indices = find_knn_in_embedding_space(
            train_embeddings, query_embedding, k=k
        )
        
        neighbor_indices = indices[0]  # Get neighbors for the single query point
        neighbor_distances = distances[0]
        
        # Highlight the query point with a prominent marker
        ax.scatter(
            test_tsne[highlight_test_idx, 0], test_tsne[highlight_test_idx, 1],
            c='red',
            s=150,
            edgecolors='darkred',
            linewidth=3,
            marker='*',
            label='Query Point (Red Star)',
            zorder=10
        )
        
        # Create KNN pie chart in the side panel
        if ax_pie is not None:
            neighbor_classes = train_labels[neighbor_indices]
            knn_description = create_knn_pie_chart(
                neighbor_classes, neighbor_distances, class_color_map, class_color_names, ax_pie, k, 10, class_names, use_semantic_names
            )
        
        # Store KNN information for metadata
        knn_info = {
            'neighbor_indices': neighbor_indices,
            'neighbor_distances': neighbor_distances,
            'neighbor_classes': neighbor_classes,
            'k': k,
            'knn_description': knn_description if ax_pie is not None else None
        }
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    
    if highlight_test_idx is not None:
        if zoom_factor > 1.0:
            ax.set_title(f't-SNE with KNN Analysis - Query Point {highlight_test_idx} (Zoom: {zoom_factor:.1f}x)')
        else:
            ax.set_title(f't-SNE with KNN Analysis - Query Point {highlight_test_idx}')
    else:
        ax.set_title('t-SNE Visualization - Training vs Test Data')
    
    # Position legend differently based on whether we have a pie chart
    if ax_pie is not None:
        ax.legend(bbox_to_anchor=(0, 1), loc='upper left')
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, alpha=0.3)
    
    # Apply layout - use GridSpec layout if we have pie chart, otherwise tight_layout
    if ax_pie is not None:
        # Manually adjust spacing for GridSpec layout with larger pie chart
        gs.update(left=0.06, right=0.99, bottom=0.08, top=0.92)
    else:
        plt.tight_layout()
    
    # Create enhanced legend text
    legend_text = create_class_legend(unique_classes, class_color_map, class_names, use_semantic_names)
    
    if knn_info and 'knn_description' in knn_info and knn_info['knn_description']:
        legend_text += f"\n\n{knn_info['knn_description']}"
    
    # Create metadata with visible classes from legend
    from .utils.styling import extract_visible_classes_from_legend
    
    metadata = {
        'n_train_points': len(train_tsne),
        'n_test_points': len(test_tsne),
        'n_classes': len(unique_classes),
        'classes': unique_classes.tolist(),
        'visible_classes': extract_visible_classes_from_legend(unique_classes, class_names, use_semantic_names),
        'highlighted_point': highlight_test_idx,
        'knn_info': knn_info,
        'has_knn_pie_chart': bool(knn_info and ax_pie is not None),
        'zoom_factor': zoom_factor if highlight_test_idx is not None else None,
        'n_visible_train': len(train_tsne_visible) if 'train_tsne_visible' in locals() else len(train_tsne),
        'n_visible_test': len(test_tsne_visible) if 'test_tsne_visible' in locals() else len(test_tsne)
    }
    
    return fig, legend_text, metadata


def create_knn_pie_chart(
    neighbor_classes: np.ndarray,
    neighbor_distances: np.ndarray,
    class_color_map: Dict,
    class_color_names: Dict,
    ax_pie: plt.Axes,
    k: int,
    max_pie_classes: int = 10,
    class_names: Optional[List[str]] = None,
    use_semantic_names: bool = False
) -> str:
    """
    Create a pie chart showing KNN class distribution with distance-based sizing.
    
    Args:
        neighbor_classes: Classes of the k nearest neighbors
        neighbor_distances: Distances to the k nearest neighbors
        class_color_map: Mapping from class labels to colors
        class_color_names: Mapping from class labels to color names
        ax_pie: Matplotlib axes for the pie chart
        k: Number of nearest neighbors
        max_pie_classes: Maximum number of classes to show in pie chart (default: 10)
        
    Returns:
        Description text for the pie chart
    """
    # Count neighbors by class
    unique_classes_knn, class_counts = np.unique(neighbor_classes, return_counts=True)
    
    # Calculate average distance for each class
    class_avg_distances = {}
    for cls in unique_classes_knn:
        mask = neighbor_classes == cls
        class_avg_distances[cls] = np.mean(neighbor_distances[mask])
    
    # Limit to top N classes by count (most represented neighbors)
    if len(unique_classes_knn) > max_pie_classes:
        # Sort by count (descending) and take top (max_pie_classes - 1) to leave room for "Other"
        sorted_indices = np.argsort(class_counts)[::-1][:max_pie_classes - 1]
        unique_classes_knn = unique_classes_knn[sorted_indices]
        class_counts = class_counts[sorted_indices]
        
        # Update class_avg_distances to only include top classes
        class_avg_distances = {cls: class_avg_distances[cls] for cls in unique_classes_knn}
        
        # Calculate count of remaining classes for "Other" category
        other_count = k - np.sum(class_counts)
        if other_count > 0:
            # Add "Other" category
            unique_classes_knn = np.append(unique_classes_knn, -1)  # Use -1 as "Other" class ID
            class_counts = np.append(class_counts, other_count)
            class_avg_distances[-1] = np.mean([class_avg_distances[cls] for cls in class_avg_distances.keys()])
    
    # Prepare data for pie chart
    colors = []
    labels = []
    for i, cls in enumerate(unique_classes_knn):
        count = class_counts[i]
        avg_dist = class_avg_distances[cls]
        
        if cls == -1:  # "Other" category
            colors.append('lightgray')
            labels.append(f'Other\n{count}/{k}\nAvgDist: {avg_dist:.1f}')
        else:
            colors.append(class_color_map[cls])
            color_name = class_color_names[cls]
            class_label = format_class_label(cls, class_names, use_semantic_names)
            labels.append(f'{class_label}\n{count}/{k}\nAvgDist: {avg_dist:.1f}')
    
    # Create pie chart with distance-based "explosion" (larger radius for closer points)
    max_dist = max(class_avg_distances.values())
    min_dist = min(class_avg_distances.values())
    
    # Calculate explosion values (closer points = larger explosion = more prominent)
    if max_dist > min_dist:
        explode = [(max_dist - class_avg_distances[cls]) / (max_dist - min_dist) * 0.1 
                  for cls in unique_classes_knn]
    else:
        explode = [0.05] * len(unique_classes_knn)  # Small uniform explosion if all distances equal
    
    # Create the pie chart with better spacing
    wedges, texts, autotexts = ax_pie.pie(
        class_counts,
        labels=labels,
        colors=colors,
        autopct='',  # We'll handle percentages in labels
        startangle=90,
        explode=explode,
        textprops={'fontsize': 7, 'weight': 'normal'},  # Smaller, normal weight text
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.5},  # Thinner edge lines
        labeldistance=1.2,  # Move labels further outward
        pctdistance=0.85    # Keep any percentage text closer to center
    )
    
    # Update title to show if classes were limited
    title = f'K-NN Distribution (k={k})'
    if len(np.unique(neighbor_classes)) > max_pie_classes:
        title += f' (Top {max_pie_classes})'
    ax_pie.set_title(title, fontsize=9, weight='bold', pad=8)
    
    # Ensure pie chart is circular and well-positioned
    ax_pie.axis('equal')
    
    # Adjust text positioning to reduce overlap
    for text in texts:
        text.set_horizontalalignment('center')
        text.set_verticalalignment('center')
    
    # Create description text
    description = f"K-NN Analysis (k={k}):\n"
    for i, cls in enumerate(unique_classes_knn):
        count = class_counts[i]
        percentage = (count / k) * 100
        avg_dist = class_avg_distances[cls]
        
        if cls == -1:  # "Other" category
            description += f"• Other: {count} neighbors ({percentage:.0f}%), AvgDist: {avg_dist:.1f}\n"
        else:
            color_name = class_color_names[cls]
            class_label = format_class_label(cls, class_names, use_semantic_names)
            description += f"• {class_label}: {count} neighbors ({percentage:.0f}%), AvgDist: {avg_dist:.1f}\n"
    
    return description.strip()




def create_tsne_3d_plot_with_knn(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_labels: np.ndarray,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    highlight_test_idx: Optional[int] = None,
    k: int = 5,
    figsize: Tuple[int, int] = (20, 15),
    viewing_angles: Optional[List[Tuple[int, int]]] = None,
    zoom_factor: float = 2.0,
    class_names: Optional[List[str]] = None,
    use_semantic_names: bool = False
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create multiple 3D t-SNE plots with KNN pie chart from different viewing angles.
    
    Args:
        train_tsne: 3D t-SNE coordinates for training data [n_train, 3]
        test_tsne: 3D t-SNE coordinates for test data [n_test, 3]
        train_labels: Training labels [n_train]
        train_embeddings: Original training embeddings for KNN [n_train, embedding_dim]
        test_embeddings: Original test embeddings for KNN [n_test, embedding_dim]
        highlight_test_idx: Index of test point to highlight and show KNN pie chart
        k: Number of nearest neighbors to analyze
        figsize: Figure size (width, height)
        viewing_angles: List of (elevation, azimuth) tuples for different views
        zoom_factor: Zoom level (2.0 = 200% zoom, showing 50% of the range)
        
    Returns:
        fig: Matplotlib figure object with multiple 3D subplots
        legend_text: Text description of the legend
        metadata: Dictionary with plot metadata
    """
    if viewing_angles is None:
        # Default viewing angles: front, side, top, and isometric views
        viewing_angles = [
            (20, 45),   # Isometric view
            (0, 0),     # Front view (XZ plane)
            (0, 90),    # Side view (YZ plane)
            (90, 0),    # Top view (XY plane)
        ]
    
    # Create subplots for multiple views plus optional pie chart
    n_views = len(viewing_angles)
    fig = plt.figure(figsize=figsize)
    
    # If highlighting a point, reserve space for KNN pie chart
    if highlight_test_idx is not None:
        # Create grid: 2x4 layout with pie chart taking 2 columns in bottom right
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1], 
                             hspace=0.15, wspace=0.15)
        ax_pie = fig.add_subplot(gs[1, 2:])  # Bottom right 2 columns for bigger pie chart
        view_positions = [(0, 0), (0, 1), (0, 2), (1, 0)]  # First 4 positions for 3D views
    else:
        # Standard 2x2 grid for 4 views
        gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.2)
        ax_pie = None
        view_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # Get unique classes and create consistent color mapping
    unique_classes = np.unique(train_labels)
    class_color_map = create_distinct_color_map(unique_classes)
    class_color_names = get_class_color_name_map(unique_classes)
    
    view_names = ['Isometric View', 'Front View (XZ)', 'Side View (YZ)', 'Top View (XY)']
    
    # Find KNN for the highlighted test point if specified
    knn_info = {}
    if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_tsne):
        from clam.models.knn_utils import find_knn_in_embedding_space
        
        query_embedding = test_embeddings[highlight_test_idx:highlight_test_idx+1]
        distances, indices = find_knn_in_embedding_space(
            train_embeddings, query_embedding, k=k
        )
        
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]
        neighbor_classes = train_labels[neighbor_indices]
        
        knn_info = {
            'neighbor_indices': neighbor_indices,
            'neighbor_distances': neighbor_distances,
            'neighbor_classes': neighbor_classes,
            'k': k
        }
    
    # Apply zoom if highlighting a test point
    if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_tsne) and zoom_factor > 1.0:
        # Get the target point coordinates
        target_point = test_tsne[highlight_test_idx]
        
        # Calculate the visible range based on zoom factor
        visible_fraction = 1.0 / zoom_factor
        
        # Get original data ranges
        all_points = np.vstack([train_tsne, test_tsne])
        x_range = np.ptp(all_points[:, 0])
        y_range = np.ptp(all_points[:, 1])
        z_range = np.ptp(all_points[:, 2])
        
        # Calculate zoom window
        x_window = x_range * visible_fraction
        y_window = y_range * visible_fraction
        z_window = z_range * visible_fraction
        
        # Set axis limits centered on target point
        x_min = target_point[0] - x_window / 2
        x_max = target_point[0] + x_window / 2
        y_min = target_point[1] - y_window / 2
        y_max = target_point[1] + y_window / 2
        z_min = target_point[2] - z_window / 2
        z_max = target_point[2] + z_window / 2
        
        # Filter points to only show those within the zoom window
        train_mask = ((train_tsne[:, 0] >= x_min) & (train_tsne[:, 0] <= x_max) & 
                     (train_tsne[:, 1] >= y_min) & (train_tsne[:, 1] <= y_max) &
                     (train_tsne[:, 2] >= z_min) & (train_tsne[:, 2] <= z_max))
        test_mask = ((test_tsne[:, 0] >= x_min) & (test_tsne[:, 0] <= x_max) & 
                    (test_tsne[:, 1] >= y_min) & (test_tsne[:, 1] <= y_max) &
                    (test_tsne[:, 2] >= z_min) & (test_tsne[:, 2] <= z_max))
        
        train_tsne_visible = train_tsne[train_mask]
        train_labels_visible = train_labels[train_mask]
        test_tsne_visible = test_tsne[test_mask]
    else:
        # No zoom, show all points
        train_tsne_visible = train_tsne
        train_labels_visible = train_labels
        test_tsne_visible = test_tsne
        x_min = x_max = y_min = y_max = z_min = z_max = None
    
    for i, (elev, azim) in enumerate(viewing_angles):
        row, col = view_positions[i]
        ax = fig.add_subplot(gs[row, col], projection='3d')
        
        # Plot training points with colors (only visible ones)
        for class_label in unique_classes:
            mask = train_labels_visible == class_label
            if np.any(mask):
                ax.scatter(
                    train_tsne_visible[mask, 0], train_tsne_visible[mask, 1], train_tsne_visible[mask, 2],
                    c=[class_color_map[class_label]], 
                    label=f'Training Class {class_label}' if i == 0 else "",
                    alpha=0.7,
                    s=40,
                    edgecolors='black',
                    linewidth=0.5
                )
        
        # Plot all test points in gray (only visible ones)
        if len(test_tsne_visible) > 0:
            ax.scatter(
                test_tsne_visible[:, 0], test_tsne_visible[:, 1], test_tsne_visible[:, 2],
                c='lightgray',
                alpha=0.6,
                s=50,
                edgecolors='gray',
                linewidth=0.8,
                marker='s',
                label='Test Points (Light Gray)' if i == 0 else ""
            )
        
        # Highlight specific test point for KNN analysis
        if knn_info:
            # Highlight the query point with a prominent marker
            ax.scatter(
                test_tsne[highlight_test_idx, 0], test_tsne[highlight_test_idx, 1], test_tsne[highlight_test_idx, 2],
                c='red',
                s=150,
                edgecolors='darkred',
                linewidth=3,
                marker='*',
                label='Query Point' if i == 0 else "",
                zorder=10
            )
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Set axis limits if zoomed
        if x_min is not None:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
        
        # Labels and title with compact semantic information to prevent overlap
        compact_labels = create_compact_axis_labels(semantic_axes_labels, max_chars_per_line=25)
        
        if compact_labels and 'X' in compact_labels:
            ax.set_xlabel(compact_labels['X'])
        else:
            ax.set_xlabel('t-SNE Dim 1')
        
        if compact_labels and 'Y' in compact_labels:
            ax.set_ylabel(compact_labels['Y'])
        else:
            ax.set_ylabel('t-SNE Dim 2')
        
        if compact_labels and 'Z' in compact_labels:
            ax.set_zlabel(compact_labels['Z'])
        else:
            ax.set_zlabel('t-SNE Dim 3')
        
        # Add bottom legend for full semantic information
        add_semantic_legend_to_plot(fig, ax, semantic_axes_labels, max_chars_per_line=60)
        
        # Use shorter view names if available
        view_name = view_names[i] if i < len(view_names) else f'View {i+1}'
        if highlight_test_idx is not None:
            if zoom_factor > 1.0:
                ax.set_title(f'{view_name} - Query Point {highlight_test_idx} + KNN (Zoom: {zoom_factor:.1f}x)')
            else:
                ax.set_title(f'{view_name} - Query Point {highlight_test_idx} + KNN')
        else:
            ax.set_title(f'{view_name}')
        
        # Only add legend to the first subplot to avoid clutter
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Create KNN pie chart if we have KNN info and ax_pie
    knn_description = None
    if knn_info and ax_pie is not None:
        knn_description = create_knn_pie_chart(
            knn_info['neighbor_classes'], knn_info['neighbor_distances'], 
            class_color_map, class_color_names, ax_pie, k, 10, class_names, use_semantic_names
        )
        # Update KNN info with the description
        knn_info['knn_description'] = knn_description
    
    # Apply layout - use GridSpec spacing if we have pie chart, otherwise tight_layout
    if ax_pie is not None:
        # Manually adjust spacing for GridSpec layout with multiple 3D plots
        gs.update(left=0.05, right=0.98, bottom=0.05, top=0.95)
    else:
        plt.tight_layout()
    
    # Create enhanced legend text
    legend_text = create_class_legend(unique_classes, class_color_map, class_names, use_semantic_names)
    legend_text += f"\n\nThis visualization shows {n_views} different viewing angles of the same 3D t-SNE space:"
    for i, (elev, azim) in enumerate(viewing_angles):
        view_name = view_names[i] if i < len(view_names) else f'View {i+1}'
        legend_text += f"\n- {view_name}: Elevation={elev}°, Azimuth={azim}°"
    
    if knn_info and 'knn_description' in knn_info and knn_info['knn_description']:
        legend_text += f"\n\n{knn_info['knn_description']}"
    
    # Create metadata with visible classes from legend
    from .utils.styling import extract_visible_classes_from_legend
    
    metadata = {
        'n_train_points': len(train_tsne),
        'n_test_points': len(test_tsne),
        'n_classes': len(unique_classes),
        'classes': unique_classes.tolist(),
        'visible_classes': extract_visible_classes_from_legend(unique_classes, class_names, use_semantic_names),
        'highlighted_point': highlight_test_idx,
        'viewing_angles': viewing_angles,
        'n_views': n_views,
        'is_3d': True,
        'knn_info': knn_info,
        'has_knn_pie_chart': bool(knn_info and ax_pie is not None),
        'zoom_factor': zoom_factor if highlight_test_idx is not None else None,
        'n_visible_train': len(train_tsne_visible) if 'train_tsne_visible' in locals() else len(train_tsne),
        'n_visible_test': len(test_tsne_visible) if 'test_tsne_visible' in locals() else len(test_tsne)
    }
    
    return fig, legend_text, metadata






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
    
    Args:
        train_embeddings: Training embeddings [n_train, embedding_dim]
        train_targets: Training target values [n_train]
        test_embeddings: Test embeddings [n_test, embedding_dim]
        test_targets: Optional test target values [n_test]
        perplexity: t-SNE perplexity parameter
        max_iter: Number of t-SNE iterations
        random_state: Random seed for reproducibility
        figsize: Figure size (width, height)
        colormap: Matplotlib colormap name for target values
        use_3d: Whether to create 3D visualization (default: False)
        
    Returns:
        train_tsne: 2D/3D t-SNE coordinates for training data [n_train, 2 or 3]
        test_tsne: 2D/3D t-SNE coordinates for test data [n_test, 2 or 3]
        fig: Matplotlib figure object
    """
    logger.info(f"Creating regression t-SNE visualization with {len(train_embeddings)} train and {len(test_embeddings)} test samples")
    
    # Combine embeddings for joint t-SNE
    combined_embeddings = np.vstack([train_embeddings, test_embeddings])
    n_train = len(train_embeddings)
    
    # Adjust perplexity if needed based on data size
    effective_perplexity = min(perplexity, (len(combined_embeddings) - 1) // 3)
    if effective_perplexity != perplexity:
        logger.warning(f"Adjusting perplexity from {perplexity} to {effective_perplexity} due to small dataset size")
    
    # Apply t-SNE
    logger.info(f"Running t-SNE with perplexity={effective_perplexity}, max_iter={max_iter}")
    tsne = TSNE(
        n_components=3 if use_3d else 2,
        perplexity=effective_perplexity,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1
    )
    
    # Suppress numerical warnings during t-SNE computation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
        tsne_results = tsne.fit_transform(combined_embeddings)
    
    # Split back into train and test
    train_tsne = tsne_results[:n_train]
    test_tsne = tsne_results[n_train:]
    
    # Create visualization
    if use_3d:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create color mapping for target values
        normalized_values, cmap, vmin, vmax = create_regression_color_map(train_targets, colormap)
        
        # Plot training points with color gradient based on target values
        scatter = ax.scatter(
            train_tsne[:, 0], train_tsne[:, 1], train_tsne[:, 2],
            c=train_targets,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidth=0.5,
            label='Training Data'
        )
        
        # Plot test points in gray
        ax.scatter(
            test_tsne[:, 0], test_tsne[:, 1], test_tsne[:, 2],
            c='lightgray',
            label='Test Points (Light Gray)',
            alpha=0.8,
            s=60,
            edgecolors='black',
            linewidth=0.8,
            marker='s'  # Square markers for test points
        )
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_zlabel('t-SNE Dimension 3')
        ax.set_title('3D t-SNE Visualization: Regression Data with Continuous Color Mapping')
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create color mapping for target values
        normalized_values, cmap, vmin, vmax = create_regression_color_map(train_targets, colormap)
        
        # Plot training points with color gradient based on target values
        scatter = ax.scatter(
            train_tsne[:, 0], train_tsne[:, 1],
            c=train_targets,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidth=0.5,
            label='Training Data'
        )
        
        # Plot test points in gray
        ax.scatter(
            test_tsne[:, 0], test_tsne[:, 1],
            c='lightgray',
            label='Test Points (Light Gray)',
            alpha=0.8,
            s=60,
            edgecolors='black',
            linewidth=0.8,
            marker='s'  # Square markers for test points
        )
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('t-SNE Visualization: Regression Data with Continuous Color Mapping')
        ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Target Value', rotation=270, labelpad=15)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    logger.info("Regression t-SNE visualization created successfully")
    return train_tsne, test_tsne, fig


def create_regression_tsne_3d_visualization(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    test_embeddings: np.ndarray,
    test_targets: Optional[np.ndarray] = None,
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42,
    figsize: Tuple[int, int] = (15, 12),
    colormap: str = 'viridis'
) -> Tuple[np.ndarray, np.ndarray, plt.Figure]:
    """
    Create 3D t-SNE visualization for regression data with continuous color mapping.
    
    Args:
        train_embeddings: Training embeddings [n_train, embedding_dim]
        train_targets: Training target values [n_train]
        test_embeddings: Test embeddings [n_test, embedding_dim]
        test_targets: Optional test target values [n_test]
        perplexity: t-SNE perplexity parameter
        max_iter: Number of t-SNE iterations
        random_state: Random seed for reproducibility
        figsize: Figure size (width, height)
        colormap: Matplotlib colormap name for target values
        
    Returns:
        train_tsne: 3D t-SNE coordinates for training data [n_train, 3]
        test_tsne: 3D t-SNE coordinates for test data [n_test, 3]
        fig: Matplotlib figure object with 3D subplots
    """
    logger.info(f"Creating 3D regression t-SNE visualization with {len(train_embeddings)} train and {len(test_embeddings)} test samples")
    
    # Combine embeddings for joint t-SNE
    combined_embeddings = np.vstack([train_embeddings, test_embeddings])
    n_train = len(train_embeddings)
    
    # Adjust perplexity if needed based on data size
    effective_perplexity = min(perplexity, (len(combined_embeddings) - 1) // 3)
    if effective_perplexity != perplexity:
        logger.warning(f"Adjusting perplexity from {perplexity} to {effective_perplexity} due to small dataset size")
    
    # Apply 3D t-SNE
    logger.info(f"Running 3D t-SNE with perplexity={effective_perplexity}, max_iter={max_iter}")
    tsne = TSNE(
        n_components=3,  # 3D instead of 2D
        perplexity=effective_perplexity,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1
    )
    
    # Suppress numerical warnings during t-SNE computation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
        tsne_results = tsne.fit_transform(combined_embeddings)
    
    # Split back into train and test
    train_tsne = tsne_results[:n_train]
    test_tsne = tsne_results[n_train:]
    
    # Create 3D visualization with multiple views
    fig = plt.figure(figsize=figsize)
    
    # Create color mapping for target values
    normalized_values, cmap, vmin, vmax = create_regression_color_map(train_targets, colormap)
    
    # Define viewing angles: Isometric, Front (XZ), Side (YZ), Top (XY)
    viewing_angles = [(30, 45), (0, 0), (0, 90), (90, 0)]
    view_titles = ['Isometric View', 'Front View (XZ)', 'Side View (YZ)', 'Top View (XY)']
    
    for i, ((elev, azim), title) in enumerate(zip(viewing_angles, view_titles)):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        # Plot training points with color gradient
        scatter = ax.scatter(
            train_tsne[:, 0], train_tsne[:, 1], train_tsne[:, 2],
            c=train_targets,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.7,
            s=30,
            edgecolors='black',
            linewidth=0.3
        )
        
        # Plot test points in gray
        ax.scatter(
            test_tsne[:, 0], test_tsne[:, 1], test_tsne[:, 2],
            c='lightgray',
            alpha=0.8,
            s=40,
            edgecolors='black',
            linewidth=0.5,
            marker='s'
        )
        
        ax.set_xlabel('t-SNE Dim 1')
        ax.set_ylabel('t-SNE Dim 2')
        ax.set_zlabel('t-SNE Dim 3')
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3)
    
    # Add a single colorbar for all subplots
    plt.tight_layout()
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Target Value', rotation=270, labelpad=15)
    
    # Add overall title
    fig.suptitle('3D t-SNE Visualization: Regression Data with Continuous Color Mapping', fontsize=14, y=0.95)
    
    logger.info("3D regression t-SNE visualization created successfully")
    return train_tsne, test_tsne, fig


def create_combined_regression_tsne_plot(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_targets: np.ndarray,
    highlight_test_idx: int = 0,
    figsize: Tuple[int, int] = (8, 6),
    zoom_factor: float = 1.0,
    colormap: str = 'RdBu_r'
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create a combined t-SNE plot highlighting a specific test point for regression.
    
    Args:
        train_tsne: Training t-SNE coordinates [n_train, 2]
        test_tsne: Test t-SNE coordinates [n_test, 2]
        train_targets: Training target values [n_train]
        highlight_test_idx: Index of test point to highlight
        figsize: Figure size
        zoom_factor: Zoom factor for the plot
        colormap: Matplotlib colormap name
        
    Returns:
        Tuple of (figure, legend_text, metadata)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create color mapping
    normalized_values, cmap, vmin, vmax = create_regression_color_map(train_targets, colormap)
    
    # Plot training points with color gradient
    scatter = ax.scatter(
        train_tsne[:, 0], train_tsne[:, 1],
        c=train_targets,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.7,
        s=50,
        edgecolors='black',
        linewidth=0.5,
        label='Training Data'
    )
    
    # Plot non-highlighted test points
    if len(test_tsne) > 1:
        other_test_mask = np.arange(len(test_tsne)) != highlight_test_idx
        ax.scatter(
            test_tsne[other_test_mask, 0], test_tsne[other_test_mask, 1],
            c='lightgray',
            alpha=0.6,
            s=60,
            edgecolors='black',
            linewidth=0.8,
            marker='s',
            label='Other Test Points'
        )
    
    # Highlight the specific test point
    ax.scatter(
        test_tsne[highlight_test_idx, 0], test_tsne[highlight_test_idx, 1],
        c='red',
        s=200,
        marker='*',
        edgecolors='darkred',
        linewidth=2,
        label='Query Point',
        zorder=5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Target Value', rotation=270, labelpad=15)
    
    # Apply zoom
    if zoom_factor != 1.0:
        query_x, query_y = test_tsne[highlight_test_idx]
        
        # Calculate current axis limits
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        
        # Calculate new limits based on zoom factor
        new_x_range = x_range / zoom_factor
        new_y_range = y_range / zoom_factor
        
        ax.set_xlim(query_x - new_x_range/2, query_x + new_x_range/2)
        ax.set_ylim(query_y - new_y_range/2, query_y + new_y_range/2)
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE: Regression Data with Query Point Highlighted')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create legend text
    legend_text = f"Legend: Training points colored by target value ({vmin:.3g} to {vmax:.3g}), test points in gray squares, query point as red star."
    
    # Create metadata
    metadata = {
        'target_range': (vmin, vmax),
        'colormap': colormap,
        'query_position': test_tsne[highlight_test_idx].tolist(),
        'zoom_factor': zoom_factor,
        'visible_classes': [],  # No classes in regression
        'plot_type': 'regression'
    }
    
    return fig, legend_text, metadata


# Placeholder functions for KNN-based regression visualizations
def create_regression_tsne_plot_with_knn(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_targets: np.ndarray,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    highlight_test_idx: int = 0,
    k: int = 5,
    figsize: Tuple[int, int] = (14, 10),
    zoom_factor: float = 2.0,
    colormap: str = 'viridis'
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create regression t-SNE plot with KNN connections showing target values.
    
    Args:
        train_tsne: 2D t-SNE coordinates for training data [n_train, 2]
        test_tsne: 2D t-SNE coordinates for test data [n_test, 2]
        train_targets: Training target values [n_train]
        train_embeddings: Original training embeddings for KNN [n_train, embedding_dim]
        test_embeddings: Original test embeddings for KNN [n_test, embedding_dim]
        highlight_test_idx: Index of test point to highlight and show KNN analysis
        k: Number of nearest neighbors to analyze
        figsize: Figure size (width, height)
        zoom_factor: Zoom level (2.0 = 200% zoom, showing 50% of the range)
        colormap: Matplotlib colormap name for target values
        
    Returns:
        fig: Matplotlib figure object
        legend_text: Text description of the legend
        metadata: Dictionary with plot metadata
    """
    # Create figure with subplot for main plot and KNN analysis panel
    if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_tsne):
        fig = plt.figure(figsize=figsize)
        # Create a grid: main plot takes ~60% width, KNN analysis takes ~40% width (doubled width)
        gs = fig.add_gridspec(1, 5, width_ratios=[2, 2, 1.5, 2, 2], hspace=0.1, wspace=0.15)
        ax = fig.add_subplot(gs[0, :3])  # Main plot spans first 3 columns
        ax_knn = fig.add_subplot(gs[0, 3:])  # KNN analysis spans last 2 columns (now wider)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_knn = None
    
    # Create color mapping
    normalized_values, cmap, vmin, vmax = create_regression_color_map(train_targets, colormap)
    
    # Apply zoom if highlighting a test point
    if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_tsne) and zoom_factor > 1.0:
        # Calculate zoom boundaries around the highlighted test point
        highlight_pos = test_tsne[highlight_test_idx]
        
        # Calculate the full range of the data
        all_x = np.concatenate([train_tsne[:, 0], test_tsne[:, 0]])
        all_y = np.concatenate([train_tsne[:, 1], test_tsne[:, 1]])
        x_range = np.max(all_x) - np.min(all_x)
        y_range = np.max(all_y) - np.min(all_y)
        
        # Calculate zoom window size (smaller window = more zoom)
        zoom_x_radius = (x_range / zoom_factor) / 2
        zoom_y_radius = (y_range / zoom_factor) / 2
        
        # Define zoom boundaries
        x_min = highlight_pos[0] - zoom_x_radius
        x_max = highlight_pos[0] + zoom_x_radius
        y_min = highlight_pos[1] - zoom_y_radius
        y_max = highlight_pos[1] + zoom_y_radius
        
        # Filter points within zoom region
        train_in_zoom = ((train_tsne[:, 0] >= x_min) & (train_tsne[:, 0] <= x_max) & 
                        (train_tsne[:, 1] >= y_min) & (train_tsne[:, 1] <= y_max))
        test_in_zoom = ((test_tsne[:, 0] >= x_min) & (test_tsne[:, 0] <= x_max) & 
                       (test_tsne[:, 1] >= y_min) & (test_tsne[:, 1] <= y_max))
        
        train_tsne_visible = train_tsne[train_in_zoom]
        train_targets_visible = train_targets[train_in_zoom]
        test_tsne_visible = test_tsne[test_in_zoom]
        
        # Set axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        # No zoom, show all points
        train_tsne_visible = train_tsne
        train_targets_visible = train_targets
        test_tsne_visible = test_tsne
    
    # Plot training points with color gradient (only visible ones)
    if len(train_tsne_visible) > 0:
        scatter = ax.scatter(
            train_tsne_visible[:, 0], train_tsne_visible[:, 1],
            c=train_targets_visible,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidth=0.5,
            label='Training Data'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Target Value', rotation=270, labelpad=15)
    
    # Plot all test points in gray (only visible ones)
    if len(test_tsne_visible) > 0:
        ax.scatter(
            test_tsne_visible[:, 0], test_tsne_visible[:, 1],
            c='lightgray',
            alpha=0.6,
            s=60,
            edgecolors='gray',
            linewidth=0.8,
            marker='s',
            label='Test Points (Light Gray)'
        )
    
    # If highlighting a specific test point, show KNN analysis
    knn_info = {}
    if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_tsne):
        # Import KNN utilities
        from clam.models.knn_utils import find_knn_in_embedding_space
        
        # Find KNN for this specific test point
        query_embedding = test_embeddings[highlight_test_idx:highlight_test_idx+1]
        distances, indices = find_knn_in_embedding_space(
            train_embeddings, query_embedding, k=k
        )
        
        neighbor_indices = indices[0]  # Get neighbors for the single query point
        neighbor_distances = distances[0]
        neighbor_targets = train_targets[neighbor_indices]
        
        # Highlight the query point with a prominent marker
        ax.scatter(
            test_tsne[highlight_test_idx, 0], test_tsne[highlight_test_idx, 1],
            c='red',
            s=150,
            edgecolors='darkred',
            linewidth=3,
            marker='*',
            label='Query Point (Red Star)',
            zorder=10
        )
        
        # Create KNN analysis in the side panel
        if ax_knn is not None:
            knn_description = create_knn_regression_analysis(
                neighbor_targets, neighbor_distances, ax_knn, k
            )
        
        # Store KNN information for metadata
        knn_info = {
            'neighbor_indices': neighbor_indices,
            'neighbor_distances': neighbor_distances,
            'neighbor_targets': neighbor_targets,
            'k': k,
            'knn_description': knn_description if ax_knn is not None else None,
            'predicted_value': np.mean(neighbor_targets)  # Simple KNN regression prediction
        }
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    
    if highlight_test_idx is not None:
        if zoom_factor > 1.0:
            ax.set_title(f't-SNE with KNN Regression Analysis - Query Point {highlight_test_idx} (Zoom: {zoom_factor:.1f}x)')
        else:
            ax.set_title(f't-SNE with KNN Regression Analysis - Query Point {highlight_test_idx}')
    else:
        ax.set_title('t-SNE Visualization - Training vs Test Data (Regression)')
    
    # Position legend differently based on whether we have a KNN panel
    if ax_knn is not None:
        ax.legend(bbox_to_anchor=(0, 1), loc='upper left')
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, alpha=0.3)
    
    # Apply layout - use GridSpec layout if we have KNN panel, otherwise tight_layout
    if ax_knn is not None:
        # Manually adjust spacing for GridSpec layout
        gs.update(left=0.06, right=0.99, bottom=0.08, top=0.92)
    else:
        plt.tight_layout()
    
    # Create legend text
    legend_text = f"Regression data with continuous color mapping (colormap: {colormap})"
    
    if knn_info and 'knn_description' in knn_info and knn_info['knn_description']:
        legend_text += f"\n\n{knn_info['knn_description']}"
    
    # Create metadata
    metadata = {
        'n_train_points': len(train_tsne),
        'n_test_points': len(test_tsne),
        'target_range': (float(np.min(train_targets)), float(np.max(train_targets))),
        'visible_classes': [],  # No classes in regression
        'highlighted_point': highlight_test_idx,
        'knn_info': knn_info,
        'zoom_factor': zoom_factor,
        'colormap': colormap,
        'query_position': test_tsne[highlight_test_idx].tolist() if highlight_test_idx is not None else None
    }
    
    return fig, legend_text, metadata


def create_combined_regression_tsne_3d_plot(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_targets: np.ndarray,
    highlight_test_idx: int = 0,
    figsize: Tuple[int, int] = (12, 9),
    viewing_angles: Optional[List[Tuple[int, int]]] = None,
    zoom_factor: float = 1.0,
    colormap: str = 'RdBu_r'
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create a combined 3D t-SNE plot highlighting a specific test point for regression.
    
    Args:
        train_tsne: Training t-SNE coordinates [n_train, 3]
        test_tsne: Test t-SNE coordinates [n_test, 3]
        train_targets: Training target values [n_train]
        highlight_test_idx: Index of test point to highlight
        figsize: Figure size
        viewing_angles: List of (elevation, azimuth) tuples for viewing angles
        zoom_factor: Zoom factor for the plot
        colormap: Matplotlib colormap name
        
    Returns:
        Tuple of (figure, legend_text, metadata)
    """
    fig = plt.figure(figsize=figsize)
    
    # Default viewing angles if not provided
    if viewing_angles is None:
        viewing_angles = [(30, 45), (0, 0), (0, 90), (90, 0)]
    
    view_titles = ['Isometric View', 'Front View (XZ)', 'Side View (YZ)', 'Top View (XY)']
    
    # Create color mapping
    normalized_values, cmap, vmin, vmax = create_regression_color_map(train_targets, colormap)
    
    for i, ((elev, azim), title) in enumerate(zip(viewing_angles, view_titles)):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        # Plot training points with color gradient
        scatter = ax.scatter(
            train_tsne[:, 0], train_tsne[:, 1], train_tsne[:, 2],
            c=train_targets,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.7,
            s=30,
            edgecolors='black',
            linewidth=0.3
        )
        
        # Plot non-highlighted test points
        if len(test_tsne) > 1:
            other_test_mask = np.arange(len(test_tsne)) != highlight_test_idx
            if np.any(other_test_mask):
                ax.scatter(
                    test_tsne[other_test_mask, 0], 
                    test_tsne[other_test_mask, 1], 
                    test_tsne[other_test_mask, 2],
                    c='lightgray',
                    alpha=0.6,
                    s=40,
                    edgecolors='black',
                    linewidth=0.5,
                    marker='s'
                )
        
        # Highlight the specific test point
        ax.scatter(
            test_tsne[highlight_test_idx, 0], 
            test_tsne[highlight_test_idx, 1], 
            test_tsne[highlight_test_idx, 2],
            c='red',
            s=200,
            marker='*',
            edgecolors='darkred',
            linewidth=2,
            zorder=5
        )
        
        ax.set_xlabel('t-SNE Dim 1')
        ax.set_ylabel('t-SNE Dim 2')
        ax.set_zlabel('t-SNE Dim 3')
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3)
    
    # Add colorbar
    plt.tight_layout()
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Target Value', rotation=270, labelpad=15)
    
    # Add overall title
    fig.suptitle('3D t-SNE: Regression Data with Query Point Highlighted', fontsize=14, y=0.95)
    
    # Create legend text
    legend_text = f"Legend: Training points colored by target value ({vmin:.3g} to {vmax:.3g}), test points in gray squares, query point as red star."
    
    # Create metadata
    metadata = {
        'target_range': (vmin, vmax),
        'visible_classes': [],  # No classes in regression
        'colormap': colormap,
        'query_position': test_tsne[highlight_test_idx].tolist(),
        'viewing_angles': viewing_angles,
        'zoom_factor': zoom_factor
    }
    
    return fig, legend_text, metadata


def create_regression_tsne_3d_plot_with_knn(
    train_tsne: np.ndarray,
    test_tsne: np.ndarray,
    train_targets: np.ndarray,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    highlight_test_idx: int = 0,
    k: int = 5,
    figsize: Tuple[int, int] = (16, 10),
    viewing_angles: Optional[List[Tuple[int, int]]] = None,
    zoom_factor: float = 1.0,
    colormap: str = 'viridis'
) -> Tuple[plt.Figure, str, Dict]:
    """
    Create 3D regression t-SNE plot with KNN connections showing target values.
    
    Args:
        train_tsne: 3D t-SNE coordinates for training data [n_train, 3]
        test_tsne: 3D t-SNE coordinates for test data [n_test, 3]
        train_targets: Training target values [n_train]
        train_embeddings: Original training embeddings for KNN [n_train, embedding_dim]
        test_embeddings: Original test embeddings for KNN [n_test, embedding_dim]
        highlight_test_idx: Index of test point to highlight and show KNN analysis
        k: Number of nearest neighbors to analyze
        figsize: Figure size (width, height)
        viewing_angles: List of (elevation, azimuth) tuples for viewing angles
        zoom_factor: Zoom level for the plot
        colormap: Matplotlib colormap name for target values
        
    Returns:
        fig: Matplotlib figure object
        legend_text: Text description of the legend
        metadata: Dictionary with plot metadata
    """
    # Create figure with subplot for main 3D plot and KNN analysis panel
    if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_tsne):
        fig = plt.figure(figsize=figsize)
        # Create a grid: main 3D plot takes ~60% width, KNN analysis takes ~40% width (doubled width)
        gs = fig.add_gridspec(1, 5, width_ratios=[2, 2, 1.5, 2, 2], hspace=0.1, wspace=0.15)
        ax = fig.add_subplot(gs[0, :3], projection='3d')  # Main 3D plot spans first 3 columns
        ax_knn = fig.add_subplot(gs[0, 3:])  # KNN analysis spans last 2 columns (now wider)
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax_knn = None
    
    # Create color mapping
    normalized_values, cmap, vmin, vmax = create_regression_color_map(train_targets, colormap)
    
    # Plot training points with color gradient
    scatter = ax.scatter(
        train_tsne[:, 0], train_tsne[:, 1], train_tsne[:, 2],
        c=train_targets,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.7,
        s=50,
        edgecolors='black',
        linewidth=0.5,
        label='Training Data'
    )
    
    # Plot all test points in gray
    ax.scatter(
        test_tsne[:, 0], test_tsne[:, 1], test_tsne[:, 2],
        c='lightgray',
        alpha=0.6,
        s=60,
        edgecolors='gray',
        linewidth=0.8,
        marker='s',
        label='Test Points (Light Gray)'
    )
    
    # If highlighting a specific test point, show KNN analysis
    knn_info = {}
    if highlight_test_idx is not None and 0 <= highlight_test_idx < len(test_tsne):
        # Import KNN utilities
        from clam.models.knn_utils import find_knn_in_embedding_space
        
        # Find KNN for this specific test point
        query_embedding = test_embeddings[highlight_test_idx:highlight_test_idx+1]
        distances, indices = find_knn_in_embedding_space(
            train_embeddings, query_embedding, k=k
        )
        
        neighbor_indices = indices[0]  # Get neighbors for the single query point
        neighbor_distances = distances[0]
        neighbor_targets = train_targets[neighbor_indices]
        
        # Highlight the query point with a prominent marker
        ax.scatter(
            test_tsne[highlight_test_idx, 0], test_tsne[highlight_test_idx, 1], test_tsne[highlight_test_idx, 2],
            c='red',
            s=150,
            edgecolors='darkred',
            linewidth=3,
            marker='*',
            label='Query Point (Red Star)',
            zorder=10
        )
        
        # Create KNN analysis in the side panel
        if ax_knn is not None:
            knn_description = create_knn_regression_analysis(
                neighbor_targets, neighbor_distances, ax_knn, k
            )
        
        # Store KNN information for metadata
        knn_info = {
            'neighbor_indices': neighbor_indices,
            'neighbor_distances': neighbor_distances,
            'neighbor_targets': neighbor_targets,
            'k': k,
            'knn_description': knn_description if ax_knn is not None else None,
            'predicted_value': np.mean(neighbor_targets)  # Simple KNN regression prediction
        }
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Target Value', rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    
    if highlight_test_idx is not None:
        ax.set_title(f'3D t-SNE with KNN Regression Analysis - Query Point {highlight_test_idx}')
    else:
        ax.set_title('3D t-SNE Visualization - Training vs Test Data (Regression)')
    
    # Set viewing angles if provided
    if viewing_angles and len(viewing_angles) > 0:
        elev, azim = viewing_angles[0]  # Use the first viewing angle
        ax.view_init(elev=elev, azim=azim)
    else:
        # Default viewing angle
        ax.view_init(elev=20, azim=45)
    
    # Position legend differently based on whether we have a KNN panel
    if ax_knn is not None:
        ax.legend(bbox_to_anchor=(0, 1), loc='upper left')
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Apply layout - use GridSpec layout if we have KNN panel, otherwise tight_layout
    if ax_knn is not None:
        # Manually adjust spacing for GridSpec layout
        gs.update(left=0.06, right=0.99, bottom=0.08, top=0.92)
    else:
        plt.tight_layout()
    
    # Create legend text
    legend_text = f"3D regression data with continuous color mapping (colormap: {colormap})"
    
    if knn_info and 'knn_description' in knn_info and knn_info['knn_description']:
        legend_text += f"\n\n{knn_info['knn_description']}"
    
    # Create metadata
    metadata = {
        'n_train_points': len(train_tsne),
        'n_test_points': len(test_tsne),
        'target_range': (float(np.min(train_targets)), float(np.max(train_targets))),
        'visible_classes': [],  # No classes in regression
        'highlighted_point': highlight_test_idx,
        'knn_info': knn_info,
        'zoom_factor': zoom_factor,
        'colormap': colormap,
        'viewing_angles': viewing_angles,
        'query_position': test_tsne[highlight_test_idx].tolist() if highlight_test_idx is not None else None
    }
    
    return fig, legend_text, metadata





# Optional: Enable new implementations by setting an environment variable
import os
_USE_NEW_IMPLEMENTATIONS = os.environ.get('CLAM_USE_NEW_TSNE', 'false').lower() == 'true'