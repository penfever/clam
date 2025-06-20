"""
CLAM Visualization Module

Enhanced visualization system for CLAM models with support for multiple
dimensionality reduction techniques, decision boundaries, pattern analysis,
and composable visual reasoning for VLM backends.
"""

from .base import BaseVisualization, VisualizationConfig, VisualizationResult
from .context.composer import ContextComposer

# Embedding visualizations
from .embeddings.tsne import TSNEVisualization
from .embeddings.umap import UMAPVisualization
from .embeddings.manifold import (
    LocallyLinearEmbeddingVisualization,
    SpectralEmbeddingVisualization,
    IsomapVisualization,
    MDSVisualization
)
from .embeddings.pca import PCAVisualization

# Decision and pattern visualizations
from .decision.regions import DecisionRegionsVisualization
from .patterns.frequent import FrequentPatternsVisualization

__all__ = [
    # Base classes
    'BaseVisualization',
    'VisualizationConfig', 
    'VisualizationResult',
    'ContextComposer',
    
    # Embedding visualizations
    'TSNEVisualization',
    'UMAPVisualization',
    'LocallyLinearEmbeddingVisualization',
    'SpectralEmbeddingVisualization',
    'IsomapVisualization',
    'MDSVisualization',
    'PCAVisualization',
    
    # Decision and pattern visualizations
    'DecisionRegionsVisualization',
    'FrequentPatternsVisualization',
]

# Version
__version__ = "1.0.0"