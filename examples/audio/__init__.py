"""
Audio classification examples using CLAM with Whisper embeddings.
"""

from .clam_tsne_audio_baseline import ClamAudioTsneClassifier
from .audio_datasets import ESC50Dataset, UrbanSound8KDataset, RAVDESSDataset

__all__ = [
    'ClamAudioTsneClassifier',
    'ESC50Dataset',
    'UrbanSound8KDataset', 
    'RAVDESSDataset'
]