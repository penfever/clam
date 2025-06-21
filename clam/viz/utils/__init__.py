"""
Visualization utilities for CLAM.

This module provides common functionality for saving and processing
visualizations across different CLAM implementations.
"""

from .common import plot_to_image, save_visualization_with_metadata, create_output_directories, generate_visualization_filename, close_figure_safely

__all__ = ['plot_to_image', 'save_visualization_with_metadata', 'create_output_directories', 'generate_visualization_filename', 'close_figure_safely']