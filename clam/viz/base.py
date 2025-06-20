"""
Base classes for the CLAM visualization system.

This module provides abstract base classes and data structures for implementing
modular visualization components that can be composed together for enhanced
reasoning in VLM backends.
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union
from PIL import Image
import logging

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    
    # General parameters
    figsize: Tuple[int, int] = (8, 6)
    dpi: int = 100
    random_state: int = 42
    
    # Color and styling
    colormap: str = 'tab10'
    point_size: float = 50.0
    alpha: float = 0.7
    
    # 3D options
    use_3d: bool = False
    viewing_angles: Optional[List[Tuple[float, float]]] = None
    
    # Zoom and layout
    zoom_factor: float = 1.0
    tight_layout: bool = True
    
    # Text and labels
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    zlabel: Optional[str] = None
    show_legend: bool = True
    
    # Output format
    image_format: str = 'RGB'
    max_image_size: int = 2048
    
    # Task-specific options
    task_type: str = 'classification'  # 'classification' or 'regression'
    
    # Additional options for specific visualizations
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationResult:
    """Result of a visualization operation."""
    
    # Core outputs
    image: Image.Image
    transformed_data: np.ndarray
    description: str
    
    # Metadata
    method_name: str
    config: VisualizationConfig
    
    # Analysis information
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    fit_time: float = 0.0
    transform_time: float = 0.0
    plot_time: float = 0.0
    
    # Coordinates for specific points (if highlighting)
    highlighted_indices: Optional[List[int]] = None
    highlighted_coords: Optional[np.ndarray] = None
    
    # Legend information for VLM prompts
    legend_text: str = ""
    
    # Quality metrics (if applicable)
    stress: Optional[float] = None  # For MDS
    reconstruction_error: Optional[float] = None  # For LLE
    explained_variance: Optional[float] = None  # For PCA


class BaseVisualization(ABC):
    """
    Abstract base class for all visualization methods.
    
    This class defines the interface that all visualization methods must implement
    to be compatible with the context composer system.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the visualization method.
        
        Args:
            config: Configuration for the visualization
        """
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # State variables
        self._fitted = False
        self._transformer = None
        self._last_result = None
        
    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the name of the visualization method."""
        pass
    
    @property
    @abstractmethod
    def supports_3d(self) -> bool:
        """Return whether this method supports 3D visualization."""
        pass
    
    @property
    @abstractmethod
    def supports_regression(self) -> bool:
        """Return whether this method supports regression tasks."""
        pass
    
    @property
    @abstractmethod
    def supports_new_data(self) -> bool:
        """Return whether this method can transform new data after fitting."""
        pass
    
    @abstractmethod
    def _create_transformer(self, **kwargs) -> Any:
        """
        Create the underlying transformer object.
        
        Returns:
            Transformer object (e.g., TSNE, UMAP, etc.)
        """
        pass
    
    @abstractmethod
    def _get_default_description(self, n_samples: int, n_features: int) -> str:
        """
        Get a default description for this visualization method.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            Description string
        """
        pass
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Fit the visualization method and transform the data.
        
        Args:
            X: Input data [n_samples, n_features]
            y: Optional target values [n_samples]
            **kwargs: Additional parameters for the method
            
        Returns:
            Transformed coordinates [n_samples, n_components]
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
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using the fitted method.
        
        Args:
            X: Input data [n_samples, n_features]
            
        Returns:
            Transformed coordinates [n_samples, n_components]
        """
        if not self._fitted:
            raise ValueError("Must call fit_transform before transform")
        
        if not self.supports_new_data:
            raise NotImplementedError(f"{self.method_name} does not support transforming new data")
        
        import time
        start_time = time.time()
        
        transformed = self._transformer.transform(X)
        
        self._last_transform_time = time.time() - start_time
        
        return transformed
    
    def generate_plot(
        self,
        transformed_data: np.ndarray,
        y: Optional[np.ndarray] = None,
        highlight_indices: Optional[List[int]] = None,
        test_data: Optional[np.ndarray] = None,
        **kwargs
    ) -> VisualizationResult:
        """
        Generate a plot from transformed data.
        
        Args:
            transformed_data: Transformed coordinates
            y: Optional target values for coloring
            highlight_indices: Indices of points to highlight
            test_data: Optional test data coordinates
            **kwargs: Additional plotting parameters
            
        Returns:
            VisualizationResult object
        """
        import time
        import io
        
        plot_start = time.time()
        
        # Determine number of components
        n_components = transformed_data.shape[1]
        use_3d = self.config.use_3d and n_components >= 3 and self.supports_3d
        
        # Create figure
        if use_3d:
            fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot based on task type
        if self.config.task_type == 'regression' and y is not None:
            plot_result = self._plot_regression(ax, transformed_data, y, highlight_indices, test_data, use_3d)
        else:
            plot_result = self._plot_classification(ax, transformed_data, y, highlight_indices, test_data, use_3d)
        
        # Apply styling
        self._apply_plot_styling(ax, use_3d)
        
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
        
        # Resize if needed
        if image.width > self.config.max_image_size or image.height > self.config.max_image_size:
            ratio = min(self.config.max_image_size / image.width, self.config.max_image_size / image.height)
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        plot_time = time.time() - plot_start
        
        # Create result
        result = VisualizationResult(
            image=image,
            transformed_data=transformed_data,
            description=self._get_default_description(len(transformed_data), transformed_data.shape[1]),
            method_name=self.method_name,
            config=self.config,
            fit_time=getattr(self, '_last_fit_time', 0.0),
            transform_time=getattr(self, '_last_transform_time', 0.0),
            plot_time=plot_time,
            highlighted_indices=highlight_indices,
            highlighted_coords=transformed_data[highlight_indices] if highlight_indices else None,
            legend_text=plot_result.get('legend_text', ''),
            metadata=plot_result.get('metadata', {})
        )
        
        # Add method-specific quality metrics
        self._add_quality_metrics(result)
        
        self._last_result = result
        return result
    
    def _plot_classification(
        self,
        ax,
        transformed_data: np.ndarray,
        y: Optional[np.ndarray],
        highlight_indices: Optional[List[int]],
        test_data: Optional[np.ndarray],
        use_3d: bool
    ) -> Dict[str, Any]:
        """Plot for classification tasks."""
        
        legend_text_parts = []
        metadata = {'classes': [], 'plot_type': 'classification'}
        
        if y is not None:
            unique_classes = np.unique(y)
            colors = plt.cm.get_cmap(self.config.colormap)(np.linspace(0, 1, len(unique_classes)))
            
            for i, cls in enumerate(unique_classes):
                mask = y == cls
                if use_3d:
                    ax.scatter(
                        transformed_data[mask, 0],
                        transformed_data[mask, 1],
                        transformed_data[mask, 2],
                        c=[colors[i]], s=self.config.point_size, alpha=self.config.alpha,
                        label=f'Class {cls}'
                    )
                else:
                    ax.scatter(
                        transformed_data[mask, 0],
                        transformed_data[mask, 1],
                        c=[colors[i]], s=self.config.point_size, alpha=self.config.alpha,
                        label=f'Class {cls}'
                    )
                
                metadata['classes'].append(cls)
                legend_text_parts.append(f"Class {cls}: {colors[i]}")
        else:
            # No labels - use single color
            if use_3d:
                ax.scatter(
                    transformed_data[:, 0],
                    transformed_data[:, 1],
                    transformed_data[:, 2],
                    s=self.config.point_size, alpha=self.config.alpha
                )
            else:
                ax.scatter(
                    transformed_data[:, 0],
                    transformed_data[:, 1],
                    s=self.config.point_size, alpha=self.config.alpha
                )
        
        # Highlight specific points
        if highlight_indices:
            if use_3d:
                ax.scatter(
                    transformed_data[highlight_indices, 0],
                    transformed_data[highlight_indices, 1],
                    transformed_data[highlight_indices, 2],
                    c='red', s=self.config.point_size * 2, alpha=1.0,
                    marker='x', linewidths=3, label='Highlighted'
                )
            else:
                ax.scatter(
                    transformed_data[highlight_indices, 0],
                    transformed_data[highlight_indices, 1],
                    c='red', s=self.config.point_size * 2, alpha=1.0,
                    marker='x', linewidths=3, label='Highlighted'
                )
            
            metadata['highlighted_indices'] = highlight_indices
        
        # Plot test data
        if test_data is not None:
            if use_3d:
                ax.scatter(
                    test_data[:, 0],
                    test_data[:, 1],
                    test_data[:, 2],
                    c='black', s=self.config.point_size * 1.5, alpha=0.8,
                    marker='^', label='Test points'
                )
            else:
                ax.scatter(
                    test_data[:, 0],
                    test_data[:, 1],
                    c='black', s=self.config.point_size * 1.5, alpha=0.8,
                    marker='^', label='Test points'
                )
        
        return {
            'legend_text': '; '.join(legend_text_parts),
            'metadata': metadata
        }
    
    def _plot_regression(
        self,
        ax,
        transformed_data: np.ndarray,
        y: np.ndarray,
        highlight_indices: Optional[List[int]],
        test_data: Optional[np.ndarray],
        use_3d: bool
    ) -> Dict[str, Any]:
        """Plot for regression tasks."""
        
        # Use continuous colormap for regression
        if use_3d:
            scatter = ax.scatter(
                transformed_data[:, 0],
                transformed_data[:, 1],
                transformed_data[:, 2],
                c=y, s=self.config.point_size, alpha=self.config.alpha,
                cmap='viridis'
            )
        else:
            scatter = ax.scatter(
                transformed_data[:, 0],
                transformed_data[:, 1],
                c=y, s=self.config.point_size, alpha=self.config.alpha,
                cmap='viridis'
            )
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Target Value')
        
        metadata = {
            'plot_type': 'regression',
            'target_min': float(np.min(y)),
            'target_max': float(np.max(y)),
            'target_mean': float(np.mean(y)),
            'target_std': float(np.std(y))
        }
        
        # Highlight specific points
        if highlight_indices:
            if use_3d:
                ax.scatter(
                    transformed_data[highlight_indices, 0],
                    transformed_data[highlight_indices, 1],
                    transformed_data[highlight_indices, 2],
                    c='red', s=self.config.point_size * 2, alpha=1.0,
                    marker='x', linewidths=3, label='Highlighted'
                )
            else:
                ax.scatter(
                    transformed_data[highlight_indices, 0],
                    transformed_data[highlight_indices, 1],
                    c='red', s=self.config.point_size * 2, alpha=1.0,
                    marker='x', linewidths=3, label='Highlighted'
                )
            
            metadata['highlighted_indices'] = highlight_indices
        
        # Plot test data
        if test_data is not None:
            if use_3d:
                ax.scatter(
                    test_data[:, 0],
                    test_data[:, 1],
                    test_data[:, 2],
                    c='black', s=self.config.point_size * 1.5, alpha=0.8,
                    marker='^', label='Test points'
                )
            else:
                ax.scatter(
                    test_data[:, 0],
                    test_data[:, 1],
                    c='black', s=self.config.point_size * 1.5, alpha=0.8,
                    marker='^', label='Test points'
                )
        
        legend_text = f"Target range: [{np.min(y):.2f}, {np.max(y):.2f}]"
        
        return {
            'legend_text': legend_text,
            'metadata': metadata
        }
    
    def _apply_plot_styling(self, ax, use_3d: bool):
        """Apply styling to the plot."""
        
        if self.config.title:
            ax.set_title(self.config.title)
        
        if self.config.xlabel:
            ax.set_xlabel(self.config.xlabel)
        else:
            ax.set_xlabel(f'{self.method_name} Component 1')
        
        if self.config.ylabel:
            ax.set_ylabel(self.config.ylabel)
        else:
            ax.set_ylabel(f'{self.method_name} Component 2')
        
        if use_3d and self.config.zlabel:
            ax.set_zlabel(self.config.zlabel)
        elif use_3d:
            ax.set_zlabel(f'{self.method_name} Component 3')
        
        if self.config.show_legend:
            ax.legend()
        
        # Apply zoom factor
        if self.config.zoom_factor != 1.0:
            if use_3d:
                # For 3D, adjust the viewing distance
                ax.dist = ax.dist * self.config.zoom_factor
            else:
                # For 2D, adjust the axis limits
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                x_center = (xlim[0] + xlim[1]) / 2
                y_center = (ylim[0] + ylim[1]) / 2
                x_range = (xlim[1] - xlim[0]) / self.config.zoom_factor
                y_range = (ylim[1] - ylim[0]) / self.config.zoom_factor
                
                ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
                ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
        
        # Apply custom viewing angles for 3D
        if use_3d and self.config.viewing_angles:
            for elev, azim in self.config.viewing_angles:
                ax.view_init(elev=elev, azim=azim)
                break  # Use first angle by default
    
    def _add_quality_metrics(self, result: VisualizationResult):
        """Add method-specific quality metrics to the result."""
        # Default implementation - subclasses can override
        pass
    
    def get_description(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> str:
        """
        Get a description of this visualization method for the given data.
        
        Args:
            X: Input data
            y: Optional target values
            
        Returns:
            Description string
        """
        base_desc = self._get_default_description(len(X), X.shape[1])
        
        if y is not None:
            if self.config.task_type == 'classification':
                n_classes = len(np.unique(y))
                base_desc += f" The data contains {n_classes} classes."
            else:
                target_range = np.max(y) - np.min(y)
                base_desc += f" The target values range from {np.min(y):.2f} to {np.max(y):.2f} (range: {target_range:.2f})."
        
        return base_desc