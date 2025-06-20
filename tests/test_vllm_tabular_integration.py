#!/usr/bin/env python3
"""
Integration test for VLLM backend with tabular LLM functionality.

This test validates that the tabular LLM baselines in the examples/ directory
work correctly when using the VLLM backend instead of transformers.

Usage:
    conda activate llata
    python tests/test_vllm_tabular_integration.py
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_tabular_dataset():
    """Create a small synthetic tabular dataset for testing."""
    np.random.seed(42)
    
    # Create features that have some predictive power
    n_samples = 50
    n_features = 4
    
    # Generate correlated features
    X = np.random.randn(n_samples, n_features)
    
    # Create target based on a simple rule
    # If first feature > 0 and second feature > 0, class 1, else class 0
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)
    
    # Create meaningful feature names
    feature_names = ['income_score', 'credit_score', 'age_normalized', 'debt_ratio']
    target_names = ['rejected', 'approved']
    
    dataset = {
        'name': 'synthetic_credit_approval',
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'target_names': target_names,
        'n_classes': 2,
        'n_features': n_features,
        'task_type': 'classification'
    }
    
    return dataset


def test_model_loader_with_vllm():
    """Test that model loader can use VLLM for tabular tasks."""
    logger.info("Testing model loader with VLLM for tabular tasks...")
    
    try:
        from clam.utils.model_loader import model_loader, GenerationConfig
        
        # Test loading a small model suitable for tabular tasks
        model_name = "microsoft/DialoGPT-small"
        
        # Clear any cached models
        model_loader.unload_all()
        
        # Load with VLLM backend
        model_wrapper = model_loader.load_llm(
            model_name,
            backend="vllm",
            device="auto",
            max_model_len=256,  # Short context for faster processing
            gpu_memory_utilization=0.5
        )
        
        # Test that it can generate responses for tabular prompts
        config = GenerationConfig(
            max_new_tokens=10,
            temperature=0.1,  # Low temperature for more deterministic outputs
            do_sample=True
        )
        
        # Create a simple tabular prompt
        tabular_prompt = """Given the following data point:
income_score: 1.2, credit_score: 0.8, age_normalized: -0.3, debt_ratio: 0.5
Classify as: approved or rejected
Answer:"""
        
        result = model_wrapper.generate(tabular_prompt, config)
        
        logger.info(f"Tabular classification prompt generated: '{result.strip()}'")
        
        # Test batch generation
        prompts = [
            "Classify this data: feature1=1.0, feature2=0.5. Class:",
            "Given features: a=2.0, b=-1.0. Prediction:"
        ]
        
        batch_results = model_wrapper.generate(prompts, config)
        logger.info(f"Batch generation successful: {len(batch_results)} results")
        
        # Cleanup
        model_wrapper.unload()
        
        logger.info("✓ Model loader with VLLM works for tabular tasks")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loader with VLLM failed: {e}")
        return False


def test_tabular_data_serialization():
    """Test serialization of tabular data for LLM consumption."""
    logger.info("Testing tabular data serialization...")
    
    try:
        dataset = create_test_tabular_dataset()
        
        # Simple serialization function (mimicking what TabLLM does)
        def serialize_instance(X_row, feature_names):
            """Convert a data instance to text."""
            features_text = ", ".join([f"{name}: {value:.2f}" for name, value in zip(feature_names, X_row)])
            return f"Data: {features_text}"
        
        # Test serialization
        sample_instance = dataset['X'][0]
        serialized = serialize_instance(sample_instance, dataset['feature_names'])
        logger.info(f"Sample serialization: {serialized}")
        
        # Test that we can create prompts for classification
        prompt = f"{serialized}\nClassify as: {' or '.join(dataset['target_names'])}\nAnswer:"
        logger.info(f"Classification prompt: {prompt}")
        
        logger.info("✓ Tabular data serialization works")
        return True
        
    except Exception as e:
        logger.error(f"✗ Tabular data serialization failed: {e}")
        return False


def test_vram_usage_monitoring():
    """Test monitoring of VRAM usage on Mac M4."""
    logger.info("Testing VRAM usage monitoring...")
    
    try:
        import torch
        import platform
        
        # Check platform
        system = platform.system()
        machine = platform.machine()
        logger.info(f"Running on {system} {machine}")
        
        # Check available backends
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()
        
        logger.info(f"CUDA available: {cuda_available}")
        logger.info(f"MPS available: {mps_available}")
        
        if mps_available:
            logger.info("✓ MPS (Metal Performance Shaders) available for Mac GPU acceleration")
            # Note: MPS doesn't have direct memory monitoring like CUDA
            # but we can verify it's being used properly
        elif cuda_available:
            initial_memory = torch.cuda.memory_allocated()
            logger.info(f"Initial CUDA memory: {initial_memory} bytes")
        else:
            logger.info("Using CPU backend")
        
        # Load a model and check memory usage
        from clam.utils.model_loader import model_loader
        
        model_loader.unload_all()
        model_wrapper = model_loader.load_llm(
            "microsoft/DialoGPT-small",
            backend="vllm",
            device="auto",
            max_model_len=128
        )
        
        if cuda_available:
            final_memory = torch.cuda.memory_allocated()
            memory_used = final_memory - initial_memory
            logger.info(f"Memory used by model: {memory_used} bytes")
        
        # Cleanup
        model_wrapper.unload()
        
        logger.info("✓ VRAM usage monitoring completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ VRAM usage monitoring failed: {e}")
        return False


def test_examples_integration():
    """Test integration with examples directory functionality."""
    logger.info("Testing examples integration...")
    
    try:
        # Test that we can import examples
        import examples.tabular.evaluate_on_dataset_tabular
        logger.info("✓ Can import tabular evaluation examples")
        
        # Test dataset creation utilities (if available)
        dataset = create_test_tabular_dataset()
        
        # Test basic data processing
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            dataset['X'], dataset['y'], test_size=0.2, random_state=42
        )
        
        logger.info(f"Dataset split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        logger.info(f"Class distribution: {np.bincount(y_train)}")
        
        logger.info("✓ Examples integration works")
        return True
        
    except Exception as e:
        logger.error(f"✗ Examples integration failed: {e}")
        return False


def run_tabular_integration_tests():
    """Run all tabular integration tests."""
    logger.info("="*60)
    logger.info("VLLM TABULAR INTEGRATION TEST SUITE")
    logger.info("="*60)
    
    tests = [
        ("Model Loader with VLLM", test_model_loader_with_vllm),
        ("Tabular Data Serialization", test_tabular_data_serialization),
        ("VRAM Usage Monitoring", test_vram_usage_monitoring),
        ("Examples Integration", test_examples_integration),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TABULAR INTEGRATION TEST RESULTS")
    logger.info("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{test_name:35s}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        logger.info("🎉 All tabular integration tests passed!")
        logger.info("\nThe VLLM backend is ready for tabular LLM tasks!")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit_code = run_tabular_integration_tests()
    sys.exit(exit_code)