"""
Context composition system for multi-visualization reasoning.
"""

from .composer import ContextComposer, CompositionConfig
from .layouts import LayoutManager, LayoutStrategy
from .prompts import PromptGenerator

__all__ = [
    'ContextComposer',
    'CompositionConfig',
    'LayoutManager', 
    'LayoutStrategy',
    'PromptGenerator'
]