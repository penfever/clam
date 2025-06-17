"""
LLATA: LLM-Augmented Tabular Adapter
A library for fine-tuning LLMs on tabular data using embeddings from tabular foundation models.
"""

__version__ = "0.1.0"

# Import main modules
from . import data
from . import models
from . import train
from . import utils

# Make submodules available directly
from .data import load_dataset, get_tabpfn_embeddings, create_llm_dataset
from .models import prepare_qwen_with_prefix_embedding
from .train import train_llm_with_tabpfn_embeddings, evaluate_llm_on_test_set
from .utils import setup_logging

__all__ = [
    'data',
    'models',
    'train',
    'utils',
    'load_dataset',
    'get_tabpfn_embeddings',
    'create_llm_dataset',
    'prepare_qwen_with_prefix_embedding',
    'train_llm_with_tabpfn_embeddings',
    'evaluate_llm_on_test_set',
    'setup_logging'
]