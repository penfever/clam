#!/usr/bin/env python3
"""
Simple test script to validate audio baseline implementations.

This script tests the WhisperKNNClassifier and CLAPZeroShotClassifier
with minimal data to ensure they work correctly.
"""

import sys
import os
import logging
import tempfile
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.audio.audio_baselines import WhisperKNNClassifier, CLAPZeroShotClassifier
from clam.utils.audio_utils import create_synthetic_audio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_data(num_samples_per_class=2, num_classes=3, duration=1.0, sample_rate=16000):
    """Create synthetic audio test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_paths = []
        labels = []
        class_names = [f"class_{i}" for i in range(num_classes)]
        
        for class_idx in range(num_classes):
            for sample_idx in range(num_samples_per_class):
                # Create synthetic audio with different frequencies for each class
                base_freq = 200 + class_idx * 100  # 200, 300, 400 Hz
                audio = create_synthetic_audio(
                    frequency=base_freq + sample_idx * 50,
                    duration=duration,
                    sample_rate=sample_rate
                )
                
                # Save to temporary file
                audio_path = Path(temp_dir) / f"class_{class_idx}_sample_{sample_idx}.wav"
                
                import soundfile as sf
                sf.write(str(audio_path), audio, sample_rate)
                
                audio_paths.append(str(audio_path))
                labels.append(class_idx)
        
        # Convert to absolute paths and copy to persistent location
        import shutil
        persistent_dir = Path("./temp_test_audio")
        persistent_dir.mkdir(exist_ok=True)
        
        persistent_paths = []
        for path in audio_paths:
            dest_path = persistent_dir / Path(path).name
            shutil.copy2(path, dest_path)
            persistent_paths.append(str(dest_path))
        
        return persistent_paths, labels, class_names


def test_whisper_knn():
    """Test Whisper KNN classifier."""
    logger.info("Testing Whisper KNN classifier...")
    
    try:
        # Create test data
        audio_paths, labels, class_names = create_test_data(num_samples_per_class=3, num_classes=2)
        
        # Split into train/test
        train_paths = audio_paths[:4]  # 2 per class
        train_labels = labels[:4]
        test_paths = audio_paths[4:]
        test_labels = labels[4:]
        
        # Initialize classifier
        classifier = WhisperKNNClassifier(
            whisper_model="tiny",  # Use tiny for speed
            n_neighbors=3,
            metric="cosine",
            weights="distance",
            standardize=True,
            device="cpu",  # Force CPU for compatibility
            seed=42
        )
        
        # Fit classifier
        logger.info(f"Fitting on {len(train_paths)} training samples...")
        classifier.fit(train_paths, train_labels, class_names)
        
        # Evaluate
        logger.info(f"Evaluating on {len(test_paths)} test samples...")
        results = classifier.evaluate(test_paths, test_labels, return_detailed=True)
        
        logger.info(f"Whisper KNN results:")
        logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        logger.info(f"  Training time: {results.get('training_time', 0):.2f}s")
        logger.info(f"  Prediction time: {results['prediction_time']:.2f}s")
        
        return True, results
        
    except Exception as e:
        logger.error(f"Whisper KNN test failed: {e}")
        return False, str(e)


def test_clap_zero_shot():
    """Test CLAP zero-shot classifier."""
    logger.info("Testing CLAP zero-shot classifier...")
    
    try:
        # Create test data
        audio_paths, labels, class_names = create_test_data(num_samples_per_class=2, num_classes=2)
        
        # Use all data for testing (zero-shot doesn't need training)
        test_paths = audio_paths
        test_labels = labels
        
        # Initialize classifier
        classifier = CLAPZeroShotClassifier(
            model_name="microsoft/msclap",
            device="cpu",  # Force CPU for compatibility
            batch_size=2,  # Small batch size
            use_amp=False  # Disable AMP for CPU
        )
        
        # "Fit" classifier (just sets up class names)
        logger.info("Setting up CLAP classifier...")
        classifier.fit([], [], class_names)  # Empty training data for zero-shot
        
        # Evaluate
        logger.info(f"Evaluating on {len(test_paths)} test samples...")
        results = classifier.evaluate(test_paths, test_labels, return_detailed=True)
        
        logger.info(f"CLAP zero-shot results:")
        logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        logger.info(f"  Training time: {results.get('training_time', 0):.2f}s")
        logger.info(f"  Prediction time: {results['prediction_time']:.2f}s")
        
        return True, results
        
    except Exception as e:
        logger.error(f"CLAP zero-shot test failed: {e}")
        return False, str(e)


def cleanup_test_data():
    """Clean up temporary test data."""
    import shutil
    test_dir = Path("./temp_test_audio")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        logger.info("Cleaned up test data")


def main():
    """Run baseline validation tests."""
    logger.info("Starting audio baselines validation...")
    
    results = {}
    
    # Test Whisper KNN
    whisper_success, whisper_result = test_whisper_knn()
    results['whisper_knn'] = {
        'success': whisper_success,
        'result': whisper_result
    }
    
    # Test CLAP zero-shot
    clap_success, clap_result = test_clap_zero_shot()
    results['clap_zero_shot'] = {
        'success': clap_success,
        'result': clap_result
    }
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("AUDIO BASELINES VALIDATION SUMMARY")
    logger.info("="*60)
    
    success_count = 0
    for model_name, result in results.items():
        if result['success']:
            success_count += 1
            if isinstance(result['result'], dict):
                accuracy = result['result'].get('accuracy', 'N/A')
                logger.info(f"{model_name:15s}: ✓ SUCCESS (accuracy: {accuracy})")
            else:
                logger.info(f"{model_name:15s}: ✓ SUCCESS")
        else:
            logger.info(f"{model_name:15s}: ✗ FAILED - {result['result']}")
    
    logger.info(f"\nValidation completed: {success_count}/{len(results)} models successful")
    
    # Cleanup
    cleanup_test_data()
    
    if success_count == len(results):
        logger.info("🎉 All audio baselines are working correctly!")
        return 0
    else:
        logger.error("❌ Some audio baselines failed validation")
        return 1


if __name__ == "__main__":
    sys.exit(main())