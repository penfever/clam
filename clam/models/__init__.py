"""
Models for CLAM.

This module includes both standard embedding approaches and vector-quantized
versions for improved efficiency and representation.
"""

from .qwen_prefix import QwenWithPrefixEmbedding, prepare_qwen_with_prefix_embedding, load_pretrained_model
from .vq import VectorQuantizer, QwenWithVQPrefixEmbedding, prepare_qwen_with_vq_prefix_embedding
from .clam_tsne import ClamTsneClassifier, ClamAudioTsneClassifier, ClamImageTsneClassifier

__all__ = [
    # Standard embedding models
    "QwenWithPrefixEmbedding", 
    "prepare_qwen_with_prefix_embedding",
    "load_pretrained_model",
    
    # Vector-quantized models
    "VectorQuantizer",
    "QwenWithVQPrefixEmbedding",
    "prepare_qwen_with_vq_prefix_embedding",
    
    # CLAM t-SNE classifiers
    "ClamTsneClassifier",
    "ClamAudioTsneClassifier",
    "ClamImageTsneClassifier"
]