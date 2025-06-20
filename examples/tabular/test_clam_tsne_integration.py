#!/usr/bin/env python
"""
Test script for the integrated CLAM t-SNE classifier with multi-visualization support.

This script tests both the legacy single-visualization mode and the new 
multi-visualization mode to ensure backward compatibility and new functionality.
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from clam.models.clam_tsne import ClamTsneClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_dataset(task_type='classification', n_samples=200, n_features=10, random_state=42):
    """Create a test dataset for validation."""
    if task_type == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features//2,
            n_redundant=n_features//4,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=random_state
        )
    else:  # regression
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features//2,
            noise=0.1,
            random_state=random_state
        )
    
    return X, y


def test_legacy_mode():
    """Test the legacy single-visualization mode for backward compatibility."""
    logger.info("Testing legacy single-visualization mode...")
    
    # Create test data
    X, y = create_test_dataset(task_type='classification', n_samples=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create classifier with legacy settings
    classifier = ClamTsneClassifier(
        modality="tabular",
        vlm_model_id="Qwen/Qwen2.5-VL-32B-Instruct",
        tsne_perplexity=15,  # Smaller for small dataset
        tsne_n_iter=300,     # Fewer iterations for testing
        enable_multi_viz=False,  # Legacy mode
        seed=42
    )
    
    # Test fitting
    try:
        classifier.fit(X_train, y_train, X_test)
        logger.info("✓ Legacy mode fitting successful")
        return True
    except Exception as e:
        logger.error(f"✗ Legacy mode fitting failed: {e}")
        return False


def test_multi_viz_mode():
    """Test the new multi-visualization mode."""
    logger.info("Testing multi-visualization mode...")
    
    # Create test data
    X, y = create_test_dataset(task_type='classification', n_samples=150)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create classifier with multi-viz settings
    classifier = ClamTsneClassifier(
        modality="tabular",
        vlm_model_id="Qwen/Qwen2.5-VL-32B-Instruct",
        enable_multi_viz=True,
        visualization_methods=['pca', 'tsne'],  # Start with basic methods
        layout_strategy='adaptive_grid',
        reasoning_focus='classification',
        multi_viz_config={
            'tsne': {'perplexity': 15, 'max_iter': 300},
            'pca': {'whiten': False}
        },
        seed=42
    )
    
    # Test fitting
    try:
        classifier.fit(X_train, y_train, X_test)
        logger.info("✓ Multi-visualization mode fitting successful")
        
        # Check that context composer was created
        if classifier.context_composer is not None:
            logger.info(f"✓ Context composer initialized with {len(classifier.context_composer.visualizations)} visualizations")
            return True
        else:
            logger.error("✗ Context composer not initialized")
            return False
    except Exception as e:
        logger.error(f"✗ Multi-visualization mode fitting failed: {e}")
        return False


def test_extended_multi_viz():
    """Test multi-visualization with more visualization methods."""
    logger.info("Testing extended multi-visualization with additional methods...")
    
    # Create test data
    X, y = create_test_dataset(task_type='classification', n_samples=120)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create classifier with extended viz methods
    visualization_methods = ['pca', 'tsne']
    
    # Try to add UMAP if available
    try:
        import umap
        visualization_methods.append('umap')
        logger.info("✓ UMAP available, adding to test")
    except ImportError:
        logger.info("ℹ UMAP not available, skipping")
    
    # Add spectral embedding
    visualization_methods.append('spectral')
    
    classifier = ClamTsneClassifier(
        modality="tabular",
        vlm_model_id="Qwen/Qwen2.5-VL-32B-Instruct", 
        enable_multi_viz=True,
        visualization_methods=visualization_methods,
        layout_strategy='hierarchical',
        reasoning_focus='comparison',
        multi_viz_config={
            'tsne': {'perplexity': 10, 'max_iter': 250},
            'pca': {'whiten': True},
            'umap': {'n_neighbors': 10, 'min_dist': 0.1},
            'spectral': {'n_neighbors': 8, 'affinity': 'nearest_neighbors'}
        },
        seed=42
    )
    
    # Test fitting
    try:
        classifier.fit(X_train, y_train, X_test)
        logger.info(f"✓ Extended multi-viz fitting successful with {len(visualization_methods)} methods")
        
        # Verify all methods were added
        if classifier.context_composer is not None:
            actual_methods = len(classifier.context_composer.visualizations)
            expected_methods = len(visualization_methods)
            if actual_methods == expected_methods:
                logger.info(f"✓ All {expected_methods} visualization methods successfully integrated")
                return True
            else:
                logger.warning(f"⚠ Expected {expected_methods} methods but got {actual_methods}")
                return False
        else:
            logger.error("✗ Context composer not initialized")
            return False
    except Exception as e:
        logger.error(f"✗ Extended multi-viz fitting failed: {e}")
        return False


def test_config_persistence():
    """Test that configuration is properly stored and retrieved."""
    logger.info("Testing configuration persistence...")
    
    # Create classifier with specific config
    classifier = ClamTsneClassifier(
        modality="tabular",
        enable_multi_viz=True,
        visualization_methods=['pca', 'tsne', 'isomap'],
        layout_strategy='focus_plus_context',
        reasoning_focus='classification',
        multi_viz_config={'test_param': 'test_value'},
        seed=123
    )
    
    # Get config
    config = classifier.get_config()
    
    # Check that new parameters are included
    expected_keys = [
        'enable_multi_viz', 'visualization_methods', 'layout_strategy',
        'reasoning_focus', 'multi_viz_config'
    ]
    
    missing_keys = [key for key in expected_keys if key not in config]
    if missing_keys:
        logger.error(f"✗ Missing config keys: {missing_keys}")
        return False
    
    # Check values
    if (config['enable_multi_viz'] == True and
        config['visualization_methods'] == ['pca', 'tsne', 'isomap'] and
        config['layout_strategy'] == 'focus_plus_context' and
        config['reasoning_focus'] == 'classification' and
        config['multi_viz_config'] == {'test_param': 'test_value'}):
        logger.info("✓ Configuration persistence test passed")
        return True
    else:
        logger.error("✗ Configuration values don't match expected")
        return False


def main():
    """Run all integration tests."""
    logger.info("CLAM t-SNE Multi-Visualization Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Legacy Mode Compatibility", test_legacy_mode),
        ("Multi-Visualization Mode", test_multi_viz_mode),
        ("Extended Multi-Visualization", test_extended_multi_viz),
        ("Configuration Persistence", test_config_persistence),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running test: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"✗ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status:8} {test_name}")
        if success:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Integration successful.")
        return 0
    else:
        logger.error("❌ Some tests failed. Integration needs fixes.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)