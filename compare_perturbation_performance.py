#!/usr/bin/env python3
"""
Performance comparison script for original vs vectorized perturbation methods.
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
        # Reduced simulation time for faster testing
        time.sleep(0.0005)  # 0.5ms per call
        return np.random.randn(X.shape[0], self.output_dim).astype(np.float32)

def mock_reduction_func(embeddings: np.ndarray) -> np.ndarray:
    """Mock dimensionality reduction function."""
    time.sleep(0.001)  # 1ms per call
    return np.random.randn(embeddings.shape[0], 2).astype(np.float32)

def create_test_dataset(n_samples: int, n_features: int) -> tuple:
    """Create test dataset."""
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 3, n_samples)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Create metadata
    columns = []
    for i, name in enumerate(feature_names):
        col = ColumnMetadata(
            name=name,
            data_type="float64",
            semantic_description=f"Test feature {i}"
        )
        columns.append(col)
    
    target_classes = []
    for i in range(3):
        target_classes.append(TargetClassMetadata(
            name=f'class_{i}',
            meaning=f'Test class {i}'
        ))
    
    metadata = DatasetMetadata(
        dataset_name="test_comparison",
        description="Test dataset for performance comparison",
        columns=columns,
        target_classes=target_classes
    )
    
    return X, y, feature_names, metadata

def run_comparison(n_samples: int, n_features: int, perturbation_samples: int = 20):
    """Run performance comparison between original and vectorized methods."""
    
    logger.info(f"\n=== Comparison: {n_samples} samples, {n_features} features, {perturbation_samples} perturbations ===")
    
    # Create test data
    X, y, feature_names, metadata = create_test_dataset(n_samples, n_features)
    baseline_reduced = np.random.randn(n_samples, 2).astype(np.float32)
    
    results = {}
    
    # Test both implementations
    for use_smart_selection in [False, True]:
        method_name = "Smart Selection" if use_smart_selection else "Original"
        logger.info(f"\nTesting {method_name} method...")
        
        # Create fresh embedding model for each test
        embedding_model = MockEmbeddingModel(n_features)
        
        # Create semantic axes computer
        computer = SemanticAxesComputer(
            method="perturbation",
            perturbation_samples=perturbation_samples,
            perturbation_strength=0.1,
            max_perturbation_dataset_size=min(n_samples, 200),
            use_smart_feature_selection=use_smart_selection,
            max_features_to_test=min(10, n_features // 2),  # Test half the features
            feature_selection_method="pca_variance"
        )
        
        # Time the computation
        start_time = time.time()
        
        try:
            semantic_axes = computer.compute_semantic_axes(
                embeddings=np.random.randn(n_samples, 128),  # Mock embeddings
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
            
            results[method_name] = {
                'duration': duration,
                'model_calls': embedding_model.transform_calls,
                'semantic_axes': semantic_axes,
                'success': True
            }
            
            logger.info(f"  Duration: {duration:.3f}s")
            logger.info(f"  Model calls: {embedding_model.transform_calls}")
            logger.info(f"  Calls per second: {embedding_model.transform_calls / duration:.1f}")
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results[method_name] = {
                'duration': float('inf'),
                'model_calls': 0,
                'semantic_axes': {},
                'success': False,
                'error': str(e)
            }
    
    # Calculate speedup
    if results['Original']['success'] and results['Smart Selection']['success']:
        speedup = results['Original']['duration'] / results['Smart Selection']['duration']
        model_call_reduction = results['Original']['model_calls'] / results['Smart Selection']['model_calls']
        logger.info(f"\nSpeedup: {speedup:.2f}x")
        logger.info(f"Model call reduction: {model_call_reduction:.2f}x")
        
        # Verify results are similar (should produce same semantic meaning)
        orig_axes = set(results['Original']['semantic_axes'].keys())
        smart_axes = set(results['Smart Selection']['semantic_axes'].keys())
        if orig_axes == smart_axes:
            logger.info("✓ Both methods produced same axis names")
        else:
            logger.warning(f"⚠ Axis mismatch: {orig_axes} vs {smart_axes}")
    
    return results

def comprehensive_comparison():
    """Run comprehensive performance comparison."""
    
    test_cases = [
        (50, 10, 10),    # Small case
        (100, 15, 15),   # Medium case 
        (150, 25, 20),   # Larger case
    ]
    
    all_results = []
    
    for n_samples, n_features, perturbation_samples in test_cases:
        try:
            results = run_comparison(n_samples, n_features, perturbation_samples)
            
            # Store results for analysis
            for method_name, result in results.items():
                all_results.append({
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'perturbation_samples': perturbation_samples,
                    'method': method_name,
                    'duration': result['duration'],
                    'model_calls': result['model_calls'],
                    'success': result['success']
                })
        except Exception as e:
            logger.error(f"Test case failed: {e}")
    
    # Summary analysis
    if all_results:
        df = pd.DataFrame(all_results)
        df_success = df[df['success'] == True]
        
        logger.info("\n=== PERFORMANCE SUMMARY ===")
        
        if len(df_success) > 0:
            # Group by test case
            for (n_samples, n_features, pert_samples), group in df_success.groupby(['n_samples', 'n_features', 'perturbation_samples']):
                logger.info(f"\nCase: {n_samples} samples, {n_features} features, {pert_samples} perturbations")
                
                orig = group[group['method'] == 'Original']
                smart = group[group['method'] == 'Smart Selection']
                
                if len(orig) > 0 and len(smart) > 0:
                    orig_time = orig['duration'].iloc[0]
                    smart_time = smart['duration'].iloc[0]
                    speedup = orig_time / smart_time if smart_time > 0 else float('inf')
                    
                    orig_calls = orig['model_calls'].iloc[0]
                    smart_calls = smart['model_calls'].iloc[0]
                    call_reduction = orig_calls / smart_calls if smart_calls > 0 else float('inf')
                    
                    logger.info(f"  Original: {orig_time:.3f}s, {orig_calls} calls")
                    logger.info(f"  Smart Selection: {smart_time:.3f}s, {smart_calls} calls")
                    logger.info(f"  Speedup: {speedup:.2f}x")
                    logger.info(f"  Model call reduction: {call_reduction:.2f}x")
    
    return all_results

if __name__ == "__main__":
    logger.info("Starting performance comparison between original and vectorized perturbation methods...")
    
    # Run comprehensive comparison
    results = comprehensive_comparison()
    
    logger.info("\nComparison complete!")