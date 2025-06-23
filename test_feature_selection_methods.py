#!/usr/bin/env python3
"""
Test different feature selection methods for semantic axes optimization.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Any, Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add clam to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from clam.utils.semantic_axes import SemanticAxesComputer
from clam.utils.metadata_loader import DatasetMetadata, ColumnMetadata, TargetClassMetadata

class MockEmbeddingModel:
    """Mock embedding model for testing."""
    
    def __init__(self, input_dim: int, output_dim: int = 128):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transform_calls = 0
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Mock transform with realistic computation time."""
        self.transform_calls += 1
        time.sleep(0.0005)  # 0.5ms per call
        return np.random.randn(X.shape[0], self.output_dim).astype(np.float32)

def mock_reduction_func(embeddings: np.ndarray) -> np.ndarray:
    """Mock dimensionality reduction function."""
    time.sleep(0.001)  # 1ms per call
    return np.random.randn(embeddings.shape[0], 2).astype(np.float32)

def create_test_dataset(n_samples: int, n_features: int) -> tuple:
    """Create test dataset with some features more informative than others."""
    
    # Create some informative features and some noise features
    n_informative = max(3, n_features // 3)
    
    # Generate informative features (correlated with class labels)
    y = np.random.randint(0, 3, n_samples)
    X_informative = np.zeros((n_samples, n_informative))
    
    for i in range(n_informative):
        # Make feature correlated with class labels
        class_means = np.random.randn(3) * 2
        for class_idx in range(3):
            mask = y == class_idx
            X_informative[mask, i] = np.random.normal(class_means[class_idx], 0.5, np.sum(mask))
    
    # Generate noise features
    n_noise = n_features - n_informative
    X_noise = np.random.randn(n_samples, n_noise) * 0.3
    
    # Combine
    X = np.hstack([X_informative, X_noise]).astype(np.float32)
    
    feature_names = [f"informative_{i}" if i < n_informative else f"noise_{i-n_informative}" 
                    for i in range(n_features)]
    
    # Create metadata
    columns = []
    for i, name in enumerate(feature_names):
        col = ColumnMetadata(
            name=name,
            data_type="float64",
            semantic_description=f"{'Informative' if i < n_informative else 'Noise'} feature {i}"
        )
        columns.append(col)
    
    target_classes = []
    for i in range(3):
        target_classes.append(TargetClassMetadata(
            name=f'class_{i}',
            meaning=f'Test class {i}'
        ))
    
    metadata = DatasetMetadata(
        dataset_name="feature_selection_test",
        description="Test dataset with informative and noise features",
        columns=columns,
        target_classes=target_classes
    )
    
    return X, y, feature_names, metadata, n_informative

def test_feature_selection_methods():
    """Test different feature selection methods."""
    
    # Test parameters
    n_samples, n_features = 100, 20
    perturbation_samples = 15
    max_features_to_test = 8
    
    X, y, feature_names, metadata, n_informative = create_test_dataset(n_samples, n_features)
    baseline_reduced = np.random.randn(n_samples, 2).astype(np.float32)
    
    # Test different selection methods
    selection_methods = ["pca_variance", "mutual_info", "f_score"]
    
    results = {}
    
    for method in selection_methods:
        logger.info(f"\n=== Testing {method} feature selection ===")
        
        # Create fresh embedding model
        embedding_model = MockEmbeddingModel(n_features)
        
        # Create computer with this selection method
        computer = SemanticAxesComputer(
            method="perturbation",
            perturbation_samples=perturbation_samples,
            perturbation_strength=0.1,
            max_perturbation_dataset_size=n_samples,
            use_smart_feature_selection=True,
            max_features_to_test=max_features_to_test,
            feature_selection_method=method
        )
        
        start_time = time.time()
        
        try:
            semantic_axes = computer.compute_semantic_axes(
                embeddings=np.random.randn(n_samples, 128),
                reduced_coords=baseline_reduced,
                labels=y,
                feature_names=feature_names,
                metadata=metadata,
                original_features=X,
                embedding_model=embedding_model,
                reduction_func=mock_reduction_func
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Check which features were selected
            selected_features = []
            for feature_name in feature_names:
                # This is a hack to get selected features - in real usage you'd add this to the API
                pass
            
            results[method] = {
                'duration': duration,
                'model_calls': embedding_model.transform_calls,
                'semantic_axes': semantic_axes,
                'success': True
            }
            
            logger.info(f"  Duration: {duration:.3f}s")
            logger.info(f"  Model calls: {embedding_model.transform_calls}")
            logger.info(f"  Semantic axes: {list(semantic_axes.keys())}")
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results[method] = {
                'duration': float('inf'),
                'model_calls': 0,
                'semantic_axes': {},
                'success': False,
                'error': str(e)
            }
    
    # Summary
    logger.info(f"\n=== Feature Selection Method Comparison ===")
    logger.info(f"Dataset: {n_samples} samples, {n_features} features ({n_informative} informative)")
    logger.info(f"Testing {max_features_to_test} features with {perturbation_samples} perturbations each")
    
    for method, result in results.items():
        if result['success']:
            logger.info(f"  {method}: {result['duration']:.3f}s, {result['model_calls']} calls")
        else:
            logger.info(f"  {method}: FAILED")
    
    return results

def test_scalability():
    """Test how the smart selection scales with increasing feature counts."""
    
    feature_counts = [10, 20, 50, 100]
    max_features_to_test = 10  # Fixed subset size
    perturbation_samples = 10  # Reduced for faster testing
    
    logger.info(f"\n=== Scalability Test: Fixed {max_features_to_test} features tested ===")
    
    results = []
    
    for n_features in feature_counts:
        logger.info(f"\nTesting {n_features} total features...")
        
        n_samples = 100
        X, y, feature_names, metadata, n_informative = create_test_dataset(n_samples, n_features)
        baseline_reduced = np.random.randn(n_samples, 2).astype(np.float32)
        
        # Test original method
        embedding_model_orig = MockEmbeddingModel(n_features)
        computer_orig = SemanticAxesComputer(
            method="perturbation",
            perturbation_samples=perturbation_samples,
            use_smart_feature_selection=False
        )
        
        start_time = time.time()
        try:
            computer_orig.compute_semantic_axes(
                embeddings=np.random.randn(n_samples, 128),
                reduced_coords=baseline_reduced,
                labels=y,
                feature_names=feature_names,
                metadata=metadata,
                original_features=X,
                embedding_model=embedding_model_orig,
                reduction_func=mock_reduction_func
            )
            orig_duration = time.time() - start_time
            orig_calls = embedding_model_orig.transform_calls
        except Exception as e:
            logger.warning(f"Original method failed for {n_features} features: {e}")
            orig_duration = float('inf')
            orig_calls = 0
        
        # Test smart selection method
        embedding_model_smart = MockEmbeddingModel(n_features)
        computer_smart = SemanticAxesComputer(
            method="perturbation",
            perturbation_samples=perturbation_samples,
            use_smart_feature_selection=True,
            max_features_to_test=max_features_to_test,
            feature_selection_method="pca_variance"
        )
        
        start_time = time.time()
        try:
            computer_smart.compute_semantic_axes(
                embeddings=np.random.randn(n_samples, 128),
                reduced_coords=baseline_reduced,
                labels=y,
                feature_names=feature_names,
                metadata=metadata,
                original_features=X,
                embedding_model=embedding_model_smart,
                reduction_func=mock_reduction_func
            )
            smart_duration = time.time() - start_time
            smart_calls = embedding_model_smart.transform_calls
        except Exception as e:
            logger.error(f"Smart method failed for {n_features} features: {e}")
            smart_duration = float('inf')
            smart_calls = 0
        
        # Calculate metrics
        speedup = orig_duration / smart_duration if smart_duration > 0 else float('inf')
        call_reduction = orig_calls / smart_calls if smart_calls > 0 else float('inf')
        
        result = {
            'n_features': n_features,
            'orig_duration': orig_duration,
            'smart_duration': smart_duration,
            'orig_calls': orig_calls,
            'smart_calls': smart_calls,
            'speedup': speedup,
            'call_reduction': call_reduction
        }
        results.append(result)
        
        logger.info(f"  Original: {orig_duration:.3f}s, {orig_calls} calls")
        logger.info(f"  Smart: {smart_duration:.3f}s, {smart_calls} calls")
        logger.info(f"  Speedup: {speedup:.2f}x, Call reduction: {call_reduction:.2f}x")
    
    # Summary
    logger.info(f"\n=== Scalability Summary ===")
    for result in results:
        logger.info(f"{result['n_features']} features: {result['speedup']:.2f}x speedup, {result['call_reduction']:.2f}x fewer calls")
    
    return results

if __name__ == "__main__":
    logger.info("Testing different feature selection methods for semantic axes optimization...")
    
    # Test different selection methods
    method_results = test_feature_selection_methods()
    
    # Test scalability
    scalability_results = test_scalability()
    
    logger.info("\nTesting complete!")