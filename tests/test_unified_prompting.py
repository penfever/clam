#!/usr/bin/env python
"""
Test script for the unified VLM prompting system.

This script tests that the enhanced VLM prompting utilities work correctly
for both single and multi-visualization scenarios.
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from clam.utils.vlm_prompting import create_classification_prompt, create_regression_prompt
from clam.models.clam_tsne import ClamTsneClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_single_viz_prompting():
    """Test single visualization prompting (legacy mode)."""
    logger.info("Testing single visualization prompting...")
    
    # Test classification prompt
    class_names = ["Class 0", "Class 1", "Class 2"]
    prompt = create_classification_prompt(
        class_names=class_names,
        modality="tabular",
        use_knn=False,
        use_3d=False,
        dataset_description="Test tabular dataset with 3 classes"
    )
    
    # Check that prompt contains expected elements
    assert "t-SNE visualization of tabular data" in prompt
    assert "Class 0" in prompt and "Class 1" in prompt and "Class 2" in prompt
    assert "Format your response as: \"Class:" in prompt  # Structured response format
    
    logger.info("✓ Single visualization classification prompt works correctly")
    
    # Test regression prompt
    target_stats = {'min': 0.0, 'max': 10.0, 'mean': 5.0, 'std': 2.0}
    prompt = create_regression_prompt(
        target_stats=target_stats,
        modality="tabular",
        dataset_description="Test regression dataset"
    )
    
    assert "between 0 and 10" in prompt
    assert "Format your response as: \"Value:" in prompt  # Structured response format
    
    logger.info("✓ Single visualization regression prompt works correctly")
    return True


def test_multi_viz_prompting():
    """Test multi-visualization prompting (new mode)."""
    logger.info("Testing multi-visualization prompting...")
    
    # Test multi-viz classification prompt
    class_names = ["Class 0", "Class 1", "Class 2"]
    multi_viz_info = [
        {'method': 'PCA', 'description': 'Principal Component Analysis'},
        {'method': 'TSNE', 'description': 't-SNE visualization'},
        {'method': 'UMAP', 'description': 'UMAP visualization'}
    ]
    
    prompt = create_classification_prompt(
        class_names=class_names,
        modality="tabular",
        dataset_description="Test multi-viz classification",
        multi_viz_info=multi_viz_info
    )
    
    # Check multi-viz specific elements
    assert "3 different visualizations (PCA, TSNE, UMAP)" in prompt
    assert "visualization methods" in prompt
    assert "multiple methods" in prompt
    assert "Format your response as: \"Class:" in prompt  # Structured response maintained
    
    logger.info("✓ Multi-visualization classification prompt works correctly")
    
    # Test multi-viz regression prompt
    target_stats = {'min': 0.0, 'max': 10.0, 'mean': 5.0, 'std': 2.0}
    prompt = create_regression_prompt(
        target_stats=target_stats,
        modality="tabular",
        dataset_description="Test multi-viz regression",
        multi_viz_info=multi_viz_info
    )
    
    assert "3 different visualizations (PCA, TSNE, UMAP)" in prompt
    assert "color patterns across all 3 visualization methods" in prompt
    assert "Format your response as: \"Value:" in prompt  # Structured response maintained
    
    logger.info("✓ Multi-visualization regression prompt works correctly")
    return True


def test_prompt_consistency():
    """Test that prompts maintain consistency between single and multi-viz modes."""
    logger.info("Testing prompt consistency...")
    
    class_names = ["Class 0", "Class 1", "Class 2"]
    
    # Single viz prompt
    single_prompt = create_classification_prompt(
        class_names=class_names,
        modality="tabular",
        dataset_description="Test dataset"
    )
    
    # Multi viz prompt
    multi_viz_info = [{'method': 'TSNE', 'description': 't-SNE visualization'}]
    multi_prompt = create_classification_prompt(
        class_names=class_names,
        modality="tabular",
        dataset_description="Test dataset",
        multi_viz_info=multi_viz_info
    )
    
    # Both should have structured response format
    assert "Format your response as: \"Class:" in single_prompt
    assert "Format your response as: \"Class:" in multi_prompt
    
    # Both should mention the same classes
    for class_name in class_names:
        assert class_name in single_prompt
        assert class_name in multi_prompt
    
    logger.info("✓ Prompt consistency maintained between modes")
    return True


def test_integrated_clam_tsne():
    """Test that the integrated CLAM t-SNE system uses unified prompting correctly."""
    logger.info("Testing integrated CLAM t-SNE with unified prompting...")
    
    # Create test data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Test multi-viz mode
    classifier = ClamTsneClassifier(
        modality="tabular",
        vlm_model_id="Qwen/Qwen2.5-VL-32B-Instruct",
        enable_multi_viz=True,
        visualization_methods=['pca', 'tsne'],
        tsne_perplexity=15,
        tsne_n_iter=300,
        seed=42
    )
    
    # Test fitting (this should use the unified prompting system internally)
    try:
        classifier.fit(X_train, y_train, X_test)
        logger.info("✓ CLAM t-SNE integration successful - multi-viz mode fitted")
        
        # Verify that context composer was created
        assert classifier.context_composer is not None
        assert len(classifier.context_composer.visualizations) == 2
        
        logger.info("✓ Context composer properly initialized")
        return True
        
    except Exception as e:
        logger.error(f"✗ CLAM t-SNE integration failed: {e}")
        return False


def test_prompt_features():
    """Test that enhanced prompts include all expected features."""
    logger.info("Testing advanced prompt features...")
    
    # Test with KNN (should work in single mode)
    class_names = ["Class 0", "Class 1", "Class 2"]
    knn_prompt = create_classification_prompt(
        class_names=class_names,
        modality="tabular",
        use_knn=True,
        knn_k=5,
        legend_text="Test legend"
    )
    
    assert "nearest neighbors" in knn_prompt
    assert "pie chart" in knn_prompt
    assert "Test legend" in knn_prompt
    
    # Test with 3D
    three_d_prompt = create_classification_prompt(
        class_names=class_names,
        modality="tabular",
        use_3d=True
    )
    
    assert "3D" in three_d_prompt
    assert "Four different views" in three_d_prompt
    
    # Test with semantic names
    semantic_names = ["Cat", "Dog", "Bird"]
    semantic_prompt = create_classification_prompt(
        class_names=semantic_names,
        modality="tabular",
        use_semantic_names=True
    )
    
    assert "Cat" in semantic_prompt and "Dog" in semantic_prompt and "Bird" in semantic_prompt
    
    logger.info("✓ Advanced prompt features working correctly")
    return True


def main():
    """Run all unified prompting tests."""
    logger.info("Unified VLM Prompting Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Single Visualization Prompting", test_single_viz_prompting),
        ("Multi-Visualization Prompting", test_multi_viz_prompting),
        ("Prompt Consistency", test_prompt_consistency),
        ("Integrated CLAM t-SNE", test_integrated_clam_tsne),
        ("Advanced Prompt Features", test_prompt_features),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running test: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"✗ Test {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
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
        logger.info("🎉 All tests passed! Unified prompting integration successful.")
        return 0
    else:
        logger.error("❌ Some tests failed. Integration needs fixes.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)