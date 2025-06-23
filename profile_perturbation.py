#!/usr/bin/env python3
"""
Profiling script for semantic axes perturbation method.
Identifies performance bottlenecks and measures current performance.
"""

import numpy as np
import pandas as pd
import time
import cProfile
import pstats
import io
from typing import Any, Dict, List, Optional
import logging
from unittest.mock import MagicMock

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add clam to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from clam.utils.semantic_axes import SemanticAxesComputer
from clam.utils.metadata_loader import DatasetMetadata, ColumnMetadata

class MockEmbeddingModel:
    """Mock embedding model for profiling."""
    
    def __init__(self, input_dim: int, output_dim: int = 128):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transform_calls = 0
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Mock transform that simulates realistic computation time."""
        self.transform_calls += 1
        # Simulate some computation time (adjust as needed)
        time.sleep(0.001)  # 1ms per call to simulate model inference
        return np.random.randn(X.shape[0], self.output_dim).astype(np.float32)

def mock_reduction_func(embeddings: np.ndarray) -> np.ndarray:
    """Mock dimensionality reduction function."""
    # Simulate t-SNE/UMAP computation time
    time.sleep(0.002)  # 2ms per call
    return np.random.randn(embeddings.shape[0], 2).astype(np.float32)

def create_synthetic_dataset(n_samples: int, n_features: int) -> tuple:
    """Create synthetic tabular dataset for profiling."""
    
    # Generate synthetic data
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 5, n_samples)
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Create mock metadata
    columns = []
    for i, name in enumerate(feature_names):
        col = ColumnMetadata(
            name=name,
            data_type="float64",
            semantic_description=f"Synthetic feature {i} for testing"
        )
        columns.append(col)
    
    # Create target classes
    from clam.utils.metadata_loader import TargetClassMetadata
    target_classes = []
    for i in range(5):
        target_classes.append(TargetClassMetadata(
            name=f'class_{i}',
            meaning=f'Synthetic class {i}'
        ))
    
    metadata = DatasetMetadata(
        dataset_name="synthetic_test",
        description="Synthetic test dataset for profiling",
        columns=columns,
        target_classes=target_classes
    )
    
    return X, y, feature_names, metadata

def profile_perturbation_method(n_samples_list: List[int], 
                              n_features_list: List[int],
                              perturbation_samples: int = 50):
    """Profile the perturbation method with different dataset sizes."""
    
    results = []
    
    for n_samples in n_samples_list:
        for n_features in n_features_list:
            logger.info(f"\nProfiling: {n_samples} samples, {n_features} features")
            
            # Create synthetic dataset
            X, y, feature_names, metadata = create_synthetic_dataset(n_samples, n_features)
            
            # Create mock baseline reduced coordinates
            baseline_reduced = np.random.randn(n_samples, 2).astype(np.float32)
            
            # Create mock embedding model
            embedding_model = MockEmbeddingModel(n_features)
            
            # Create semantic axes computer
            computer = SemanticAxesComputer(
                method="perturbation",
                perturbation_samples=perturbation_samples,
                perturbation_strength=0.1,
                max_perturbation_dataset_size=min(n_samples, 200)  # Respect current limit
            )
            
            # Time the computation
            start_time = time.time()
            
            try:
                semantic_axes = computer._compute_perturbation_based_axes(
                    original_features=X,
                    baseline_reduced=baseline_reduced,
                    feature_names=feature_names,
                    feature_descriptions={name: name for name in feature_names},
                    axis_names=["X", "Y"],
                    embedding_model=embedding_model,
                    reduction_func=mock_reduction_func
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Calculate derived metrics
                total_operations = min(n_samples, 200) * n_features * perturbation_samples
                effective_samples = min(n_samples, 200)
                
                result = {
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'effective_samples': effective_samples,
                    'perturbation_samples': perturbation_samples,
                    'duration_seconds': duration,
                    'model_calls': embedding_model.transform_calls,
                    'total_operations': total_operations,
                    'operations_per_second': total_operations / duration if duration > 0 else 0,
                    'seconds_per_feature': duration / n_features if n_features > 0 else 0,
                    'success': True
                }
                
                logger.info(f"  Duration: {duration:.2f}s")
                logger.info(f"  Model calls: {embedding_model.transform_calls}")
                logger.info(f"  Operations per second: {result['operations_per_second']:.1f}")
                logger.info(f"  Seconds per feature: {result['seconds_per_feature']:.3f}")
                
            except Exception as e:
                logger.error(f"  Failed: {e}")
                result = {
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'effective_samples': min(n_samples, 200),
                    'perturbation_samples': perturbation_samples,
                    'duration_seconds': float('inf'),
                    'model_calls': 0,
                    'total_operations': 0,
                    'operations_per_second': 0,
                    'seconds_per_feature': float('inf'),
                    'success': False,
                    'error': str(e)
                }
            
            results.append(result)
    
    return results

def detailed_profile_single_case():
    """Run detailed profiling on a single case using cProfile."""
    
    logger.info("\n=== Detailed Profiling with cProfile ===")
    
    # Create test case
    n_samples, n_features = 100, 20
    X, y, feature_names, metadata = create_synthetic_dataset(n_samples, n_features)
    baseline_reduced = np.random.randn(n_samples, 2).astype(np.float32)
    embedding_model = MockEmbeddingModel(n_features)
    
    computer = SemanticAxesComputer(
        method="perturbation",
        perturbation_samples=10,  # Reduced for detailed profiling
        perturbation_strength=0.1,
        max_perturbation_dataset_size=100
    )
    
    # Profile with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        semantic_axes = computer._compute_perturbation_based_axes(
            original_features=X,
            baseline_reduced=baseline_reduced,
            feature_names=feature_names,
            feature_descriptions={name: name for name in feature_names},
            axis_names=["X", "Y"],
            embedding_model=embedding_model,
            reduction_func=mock_reduction_func
        )
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
    finally:
        profiler.disable()
    
    # Analyze results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    logger.info("Top functions by cumulative time:")
    logger.info(s.getvalue())
    
    return s.getvalue()

if __name__ == "__main__":
    logger.info("Starting perturbation method profiling...")
    
    # Test different dataset sizes
    n_samples_list = [50, 100, 200]  # Respect max_perturbation_dataset_size
    n_features_list = [10, 20, 50]
    
    # Basic performance profiling
    results = profile_perturbation_method(n_samples_list, n_features_list, perturbation_samples=20)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    logger.info("\n=== Performance Summary ===")
    logger.info(f"\nResults DataFrame:\n{df}")
    
    # Analyze scaling
    if len(df) > 1:
        logger.info("\n=== Scaling Analysis ===")
        
        # Group by features to see sample scaling
        for n_features in df['n_features'].unique():
            subset = df[df['n_features'] == n_features].sort_values('effective_samples')
            if len(subset) > 1:
                logger.info(f"\nFeatures={n_features}:")
                for _, row in subset.iterrows():
                    logger.info(f"  Samples={row['effective_samples']}: {row['duration_seconds']:.2f}s")
        
        # Group by samples to see feature scaling  
        for n_samples in df['effective_samples'].unique():
            subset = df[df['effective_samples'] == n_samples].sort_values('n_features')
            if len(subset) > 1:
                logger.info(f"\nSamples={n_samples}:")
                for _, row in subset.iterrows():
                    logger.info(f"  Features={row['n_features']}: {row['duration_seconds']:.2f}s")
    
    # Detailed profiling
    detailed_profile_single_case()
    
    logger.info("\nProfiling complete! Check the output above for bottlenecks.")