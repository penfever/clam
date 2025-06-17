#!/usr/bin/env python3
"""
Minimal test to validate audio baseline with synthetic data.
"""

import sys
import os
import tempfile
import numpy as np
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.audio.clam_tsne_audio_baseline import ClamAudioTsneClassifier
from clam.utils.audio_utils import create_synthetic_audio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_audio_files():
    """Create a few synthetic audio files for testing."""
    temp_dir = Path("./temp_minimal_test")
    temp_dir.mkdir(exist_ok=True)
    
    audio_paths = []
    labels = []
    
    # Create 4 files: 2 classes, 2 samples each
    for class_idx in range(2):
        for sample_idx in range(2):
            # Create synthetic audio
            frequency = 200 + class_idx * 200  # 200Hz, 400Hz
            audio = create_synthetic_audio(
                frequency=frequency,
                duration=1.0,
                sample_rate=16000
            )
            
            # Save to file
            import soundfile as sf
            audio_path = temp_dir / f"class_{class_idx}_sample_{sample_idx}.wav"
            sf.write(str(audio_path), audio, 16000)
            
            audio_paths.append(str(audio_path))
            labels.append(class_idx)
    
    return audio_paths, labels, ["class_0", "class_1"]

def test_minimal_audio():
    """Test LLATA audio classifier with minimal synthetic data."""
    logger.info("Creating synthetic test audio...")
    audio_paths, labels, class_names = create_test_audio_files()
    
    # Split into train/test
    train_paths = audio_paths[:2]  # 1 per class
    train_labels = labels[:2]
    test_paths = audio_paths[2:]   # 1 per class for testing
    test_labels = labels[2:]
    
    logger.info(f"Train: {len(train_paths)} samples, Test: {len(test_paths)} samples")
    
    try:
        # Initialize classifier with minimal settings
        classifier = ClamAudioTsneClassifier(
            whisper_model="tiny",  # Fastest model
            embedding_layer="encoder_last",
            tsne_perplexity=1.0,  # Very small for 2 points
            tsne_n_iter=250,      # Minimum
            vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
            use_3d_tsne=False,
            use_knn_connections=True,  # Test the KNN fix
            knn_k=1,              # Only 1 neighbor available
            max_vlm_image_size=512,
            tsne_zoom_factor=2.0,
            use_pca_backend=False,
            include_spectrogram=False,
            audio_duration=1.0,
            cache_dir="./temp_cache",
            device="cpu",  # Force CPU for compatibility
            seed=42
        )
        
        # Fit classifier
        logger.info("Fitting classifier...")
        classifier.fit(train_paths, train_labels, test_paths, class_names)
        
        # Test prediction on just one sample
        logger.info("Testing prediction...")
        prediction = classifier.predict([test_paths[0]])
        
        logger.info(f"Prediction successful: {prediction}")
        
        # Cleanup
        import shutil
        shutil.rmtree("./temp_minimal_test", ignore_errors=True)
        shutil.rmtree("./temp_cache", ignore_errors=True)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        # Cleanup
        import shutil
        shutil.rmtree("./temp_minimal_test", ignore_errors=True)
        shutil.rmtree("./temp_cache", ignore_errors=True)
        return False

if __name__ == "__main__":
    if test_minimal_audio():
        print("\n🎉 Minimal audio test passed!")
    else:
        print("\n❌ Minimal audio test failed!")
        sys.exit(1)