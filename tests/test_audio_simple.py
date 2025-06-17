#!/usr/bin/env python3
"""
Simple test to verify audio processing pipeline without VLM.
Tests Whisper embedding extraction and t-SNE visualization.
"""

import sys
import os
import logging
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_whisper_embeddings():
    """Test Whisper embedding extraction."""
    logger.info("Testing Whisper embedding extraction...")
    
    try:
        from clam.data.audio_embeddings import load_whisper_model, get_whisper_embeddings
        from examples.audio.audio_datasets import ESC50Dataset
        
        # Load small dataset
        dataset = ESC50Dataset("./esc50_test_data", download=True)
        paths, labels, class_names = dataset.get_samples()
        
        # Use just 10 samples for testing
        test_paths = paths[:10]
        test_labels = labels[:10]
        
        logger.info(f"Testing with {len(test_paths)} audio samples")
        
        # Extract embeddings with tiny model
        embeddings = get_whisper_embeddings(
            test_paths,
            model_name="tiny",
            cache_dir="./cache",
            device="cpu"
        )
        
        logger.info(f"✓ Whisper embeddings extracted: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Whisper embedding test failed: {e}")
        return False


def test_tsne_visualization():
    """Test t-SNE visualization creation."""
    logger.info("Testing t-SNE visualization...")
    
    try:
        from clam.data.tsne_visualization import create_tsne_visualization
        
        # Create dummy embeddings
        n_samples = 20
        embedding_dim = 384  # Whisper tiny embedding size
        
        train_embeddings = np.random.randn(n_samples, embedding_dim)
        train_labels = np.random.randint(0, 5, n_samples)  # 5 classes
        test_embeddings = np.random.randn(2, embedding_dim)
        
        # Create visualization
        train_tsne, test_tsne, fig = create_tsne_visualization(
            train_embeddings, train_labels, test_embeddings,
            perplexity=min(10, n_samples // 3),
            n_iter=250,  # Minimum allowed by sklearn
            figsize=(8, 6)
        )
        
        logger.info(f"✓ t-SNE visualization created: train={train_tsne.shape}, test={test_tsne.shape}")
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ t-SNE visualization test failed: {e}")
        return False


def test_audio_utils():
    """Test audio utility functions."""
    logger.info("Testing audio utilities...")
    
    try:
        from clam.utils.audio_utils import create_spectrogram, load_audio
        
        # Create dummy audio signal
        sr = 16000
        duration = 2  # seconds
        audio = np.random.randn(sr * duration) * 0.1  # Low amplitude
        
        # Test spectrogram creation
        spec = create_spectrogram(audio, sr, n_mels=128, db_scale=True)
        
        logger.info(f"✓ Spectrogram created: {spec.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Audio utilities test failed: {e}")
        return False


def test_dataset_loading():
    """Test dataset loading."""
    logger.info("Testing dataset loading...")
    
    try:
        from examples.audio.audio_datasets import ESC50Dataset
        
        dataset = ESC50Dataset("./esc50_test_data", download=True)
        
        # Test few-shot split
        splits = dataset.create_few_shot_split(k_shot=2, test_size=0.1, random_state=42)
        
        train_paths, train_labels = splits['train']
        test_paths, test_labels = splits['test']
        class_names = splits['class_names']
        
        logger.info(f"✓ Dataset loaded: {len(train_paths)} train, {len(test_paths)} test samples")
        logger.info(f"✓ Classes: {len(class_names)} - {', '.join(class_names[:5])}...")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Dataset loading test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🎵 Starting CLAM Audio Pipeline Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Audio Utilities", test_audio_utils),
        ("t-SNE Visualization", test_tsne_visualization),
        ("Whisper Embeddings", test_whisper_embeddings),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info("=" * 50)
    logger.info("📊 TEST RESULTS:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\n🏆 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All tests passed! Audio pipeline is working correctly.")
        logger.info("Ready to run full tests with:")
        logger.info("  python examples/audio/test_esc50.py --quick_test")
    else:
        logger.info("⚠️  Some tests failed. Check the errors above.")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)