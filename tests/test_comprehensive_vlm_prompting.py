#!/usr/bin/env python
"""
Comprehensive VLM Prompting Test and Demo

This script generates 20 prompts and responses using a small QwenVL model
across various single-viz and multi-viz permutations. It saves all inputs,
outputs, and visualizations for manual inspection.

Usage:
    python tests/test_comprehensive_vlm_prompting.py --output_dir ./test_outputs --dataset_id 31
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clam.models.clam_tsne import ClamTsneClassifier
from clam.utils.vlm_prompting import create_classification_prompt, create_regression_prompt
from clam.utils.unified_metrics import MetricsLogger
from clam.utils.json_utils import safe_json_dump
from clam.utils.class_name_utils import extract_class_names_from_labels
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLMPromptingTestSuite:
    """Comprehensive test suite for VLM prompting with various configurations."""
    
    def __init__(self, output_dir: str, task_id: int = 23, vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct", 
                 num_tests: int = None, num_samples_per_test: int = 10, backend: str = "auto", zoom_factor: float = 6.5):
        """
        Initialize the test suite.
        
        Args:
            output_dir: Directory to save outputs
            task_id: OpenML task ID (default: 23 = cmc contraceptive method choice)
            vlm_model: VLM model to use (default: small Qwen model)
            num_tests: Number of test configurations to run (default: None, runs all available)
            num_samples_per_test: Number of test samples per configuration (default: 10)
            backend: Backend to use for VLM inference (default: auto)
            zoom_factor: Zoom factor for t-SNE visualizations (default: 6.5)
        """
        self.output_dir = Path(output_dir)
        self.task_id = task_id
        self.vlm_model = vlm_model
        self.num_tests = num_tests
        self.num_samples_per_test = num_samples_per_test
        self.backend = backend
        self.zoom_factor = zoom_factor
        
        # Create output directories with nested structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "prompts").mkdir(exist_ok=True)
        (self.output_dir / "responses").mkdir(exist_ok=True)
        (self.output_dir / "configs").mkdir(exist_ok=True)
        (self.output_dir / "ground_truth").mkdir(exist_ok=True)
        
        # Fixed test indices for consistency across all tests
        self.test_indices = None
        
        # Store test results
        self.test_results = []
        
        # Number of responses per test (use the provided parameter)
        self.responses_per_test = num_samples_per_test
        
        # Load dataset
        self.X, self.y, self.dataset_info = self._load_dataset()
        
        logger.info(f"Initialized test suite with task {task_id}")
        logger.info(f"Dataset shape: {self.X.shape}, Classes: {len(np.unique(self.y))}")
    
    def _load_semantic_class_names(self, task_id: int, num_classes: int) -> Optional[List[str]]:
        """
        Load semantic class names from CC18 semantic data directory.
        
        Args:
            task_id: OpenML task ID
            num_classes: Number of classes in the dataset
            
        Returns:
            List of semantic class names if found, None otherwise
        """
        # Try to find semantic file using the general search
        try:
            from clam.utils.metadata_loader import get_metadata_loader
            loader = get_metadata_loader()
            semantic_file = loader.detect_metadata_file(task_id)
        except Exception as e:
            logger.debug(f"Could not use metadata loader: {e}")
            # Fallback to hardcoded path
            semantic_file = Path(__file__).parent.parent / "data" / "cc18_semantic" / f"{task_id}.json"
        
        if not semantic_file or not semantic_file.exists():
            logger.info(f"No semantic file found for task {task_id}")
            return None
        
        try:
            with open(semantic_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Try different structures to extract class names
            class_names = None
            
            # Method 1: target_values (maps numeric labels to semantic names)
            if 'target_values' in data:
                target_values = data['target_values']
                if isinstance(target_values, dict):
                    # Sort by key (handle both numeric and string keys)
                    try:
                        sorted_items = sorted(target_values.items(), key=lambda x: int(x[0]))
                    except ValueError:
                        # If keys are strings, sort lexicographically
                        sorted_items = sorted(target_values.items(), key=lambda x: x[0])
                    class_names = [item[1] for item in sorted_items]
            
            # Method 2: target_classes (list with name/meaning)
            if class_names is None and 'target_classes' in data:
                target_classes = data['target_classes']
                if isinstance(target_classes, list):
                    # Use 'meaning' if available, otherwise 'name'
                    class_names = []
                    for tc in target_classes:
                        if isinstance(tc, dict):
                            name = tc.get('meaning', tc.get('name', ''))
                            class_names.append(name)
                        else:
                            class_names.append(str(tc))
            
            # Method 3: instances_per_class keys
            if class_names is None and 'instances_per_class' in data:
                instances_per_class = data['instances_per_class']
                if isinstance(instances_per_class, dict):
                    class_names = list(instances_per_class.keys())
            
            # Validate and truncate to match number of classes
            if class_names:
                # Clean up class names (remove extra whitespace, etc.)
                class_names = [name.strip() for name in class_names if name.strip()]
                
                # Truncate to match actual number of classes
                if len(class_names) >= num_classes:
                    class_names = class_names[:num_classes]
                    logger.info(f"Loaded {len(class_names)} semantic class names for task {task_id}: {class_names}")
                    return class_names
                else:
                    logger.warning(f"Found {len(class_names)} semantic names but need {num_classes} for task {task_id}")
            
        except Exception as e:
            logger.warning(f"Failed to load semantic file for task {task_id}: {e}")
        
        return None
    
    def _calculate_metrics(self, prediction_details, ground_truth_labels, config_name):
        """Calculate comprehensive metrics from prediction details."""
        if not prediction_details:
            return None
        
        # Extract predictions and true labels
        predictions = []
        true_labels = []
        
        for detail in prediction_details:
            if 'parsed_prediction' in detail and 'true_label' in detail:
                try:
                    pred = int(detail['parsed_prediction'])
                    true = int(detail['true_label'])
                    predictions.append(pred)
                    true_labels.append(true)
                except (ValueError, TypeError):
                    # Skip invalid predictions
                    continue
        
        if len(predictions) == 0:
            return None
        
        # Calculate basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        balanced_acc = balanced_accuracy_score(true_labels, predictions)
        
        # Calculate precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )
        
        # Calculate per-class metrics
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            true_labels, predictions, average='micro', zero_division=0
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Completion rate
        completion_rate = len(predictions) / len(prediction_details)
        
        return {
            'config_name': config_name,
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'precision_macro': float(precision),
            'recall_macro': float(recall), 
            'f1_macro': float(f1),
            'precision_micro': float(precision_micro),
            'recall_micro': float(recall_micro),
            'f1_micro': float(f1_micro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'confusion_matrix': cm.tolist() if cm is not None else [],
            'completion_rate': float(completion_rate),
            'num_test_samples': len(predictions),
            'num_classes': len(np.unique(true_labels)),
            'support': support.tolist() if support is not None else []
        }
    
    def _load_dataset(self):
        """Load OpenML dataset with robust error handling."""
        try:
            # Load dataset from OpenML
            dataset = fetch_openml(data_id=self.task_id, as_frame=True, parser='auto')
            X = dataset.data
            y = dataset.target
            
            # Handle categorical targets
            if hasattr(y, 'cat'):
                y = y.cat.codes
            elif y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            # Convert to numpy with proper data type handling
            if hasattr(X, 'values'):
                X = X.values
            if hasattr(y, 'values'):
                y = y.values
            
            # Ensure X is numeric and handle mixed data types
            from sklearn.preprocessing import LabelEncoder
            X_processed = []
            for col in range(X.shape[1]):
                col_data = X[:, col]
                if col_data.dtype == 'object' or col_data.dtype.kind in 'SU':  # String/Unicode
                    # Encode categorical columns
                    le = LabelEncoder()
                    col_data = le.fit_transform(col_data.astype(str))
                else:
                    # Convert to float and handle any remaining issues
                    col_data = pd.to_numeric(col_data, errors='coerce')
                X_processed.append(col_data)
            
            X = np.column_stack(X_processed).astype(float)
            y = y.astype(int)
                
            # Handle missing values (now safe since everything is numeric)
            if np.isnan(X).any():
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            
            # Limit size for testing
            if len(X) > 500:
                indices = np.random.choice(len(X), 500, replace=False)
                X = X[indices]
                y = y[indices]
            
            # Extract feature names from OpenML dataset
            feature_names = None
            if hasattr(dataset, 'feature_names') and dataset.feature_names is not None:
                feature_names = list(dataset.feature_names)
            elif hasattr(dataset, 'data') and hasattr(dataset.data, 'columns'):
                feature_names = list(dataset.data.columns)
            
            dataset_info = {
                'name': dataset.DESCR if hasattr(dataset, 'DESCR') else f'OpenML_Task_{self.task_id}',
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)),
                'task_id': self.task_id,  # OpenML task ID for metadata loading
                'data_source': 'openml',
                'feature_names': feature_names
            }
            
            logger.info(f"Successfully loaded OpenML dataset {self.task_id}")
            return X, y, dataset_info
            
        except Exception as e:
            logger.warning(f"Failed to load OpenML dataset {self.task_id}: {e}")
            logger.info("Using synthetic dataset as fallback")
            
            # Fallback to synthetic data
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=300, n_features=10, n_classes=3, 
                                     n_informative=6, random_state=42)
            dataset_info = {
                'name': 'synthetic_fallback',
                'n_samples': len(X),
                'n_features': X.shape[1], 
                'n_classes': len(np.unique(y)),
                'task_id': None,
                'data_source': 'synthetic',
                'feature_names': [f'feature_{i}' for i in range(X.shape[1])]
            }
            return X, y, dataset_info
    
    def _get_test_configurations(self) -> List[Dict[str, Any]]:
        """Get comprehensive list of test configurations."""
        configs = []
        
        # Single visualization configurations
        single_viz_configs = [
            # Basic t-SNE
            {
                'name': 'basic_tsne',
                'enable_multi_viz': False,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # t-SNE with KNN
            {
                'name': 'tsne_knn',
                'enable_multi_viz': False,
                'use_3d_tsne': False,
                'use_knn_connections': True,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # 3D t-SNE
            {
                'name': 'tsne_3d',
                'enable_multi_viz': False,
                'use_3d_tsne': True,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # 3D t-SNE with KNN
            {
                'name': 'tsne_3d_knn',
                'enable_multi_viz': False,
                'use_3d_tsne': True,
                'use_knn_connections': True,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Different perplexity
            {
                'name': 'tsne_high_perplexity',
                'enable_multi_viz': False,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 25
            }
        ]
        
        # Multi-visualization configurations
        multi_viz_configs = [
            # Basic multi-viz (PCA + t-SNE)
            {
                'name': 'multi_pca_tsne',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne'],
                'layout_strategy': 'sequential',
                'nn_k': 30,
                'reasoning_focus': 'comparison'
            },
            # Three methods
            {
                'name': 'multi_pca_tsne_spectral',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne', 'spectral'],
                'layout_strategy': 'sequential',
                'nn_k': 30,
                'reasoning_focus': 'consensus'
            },
            # Linear vs nonlinear focus
            {
                'name': 'multi_linear_nonlinear',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne', 'isomap'],
                'layout_strategy': 'sequential',
                'nn_k': 30,
                'reasoning_focus': 'divergence'
            },
            # Local vs global methods
            {
                'name': 'multi_local_global',
                'enable_multi_viz': True,
                'visualization_methods': ['tsne', 'isomap', 'mds'],
                'layout_strategy': 'sequential',
                'nn_k': 30,
                'reasoning_focus': 'comparison'
            },
            # Comprehensive multi-viz
            {
                'name': 'multi_comprehensive',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne', 'spectral', 'isomap'],
                'layout_strategy': 'sequential',
                'nn_k': 30,
                'reasoning_focus': 'consensus'
            },
            # Different layout strategies
            {
                'name': 'multi_grid_layout',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne'],
                'layout_strategy': 'grid',
                'nn_k': 30,
                'reasoning_focus': 'comparison'
            },
            # With UMAP if available
            {
                'name': 'multi_with_umap',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne', 'umap'],
                'layout_strategy': 'sequential',
                'nn_k': 30,
                'reasoning_focus': 'comparison'
            }
        ]
        
        # Semantic naming variations
        semantic_configs = [
            # Single viz with semantic names - loaded from CC18 semantic data
            {
                'name': 'tsne_semantic',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,  # Load from CC18 semantic directory
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Multi-viz with semantic names
            {
                'name': 'multi_semantic',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne'],
                'layout_strategy': 'sequential',
                'reasoning_focus': 'comparison',
                'use_semantic_names': True,
                'nn_k': 30,
                'load_semantic_from_cc18': True  # Load from CC18 semantic directory
            },
            # Semantic class names test - single visualization with semantic names only
            {
                'name': 'tsne_semantic',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Semantic axes test - single visualization with axes interpretation
            {
                'name': 'tsne_semantic_axes',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'semantic_axes': True,  # NEW: Enable semantic axes computation
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Use metadata test - incorporate metadata into prompts
            {
                'name': 'tsne_use_metadata',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'use_metadata': True,  # NEW: Enable metadata incorporation
                'auto_load_metadata': True,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Combined new features test
            {
                'name': 'tsne_semantic_metadata_combined',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'semantic_axes': True,  # NEW: Enable semantic axes
                'use_metadata': True,   # NEW: Enable metadata
                'auto_load_metadata': True,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Multi-viz with semantic names only
            {
                'name': 'multi_tsne_semantic',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne'],
                'layout_strategy': 'sequential',
                'reasoning_focus': 'comparison',
                'nn_k': 30,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True
            },
            # Multi-viz with new features
            {
                'name': 'multi_semantic_axes_metadata',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne'],
                'layout_strategy': 'sequential',
                'reasoning_focus': 'comparison',
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'nn_k': 30,
                'semantic_axes': True,  # NEW: Enable semantic axes
                'use_metadata': True,   # NEW: Enable metadata
                'auto_load_metadata': True
            },
            # Perturbation semantic axes test - single visualization with perturbation method
            {
                'name': 'tsne_perturbation_axes',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'semantic_axes': True,
                'semantic_axes_method': 'perturbation',  # NEW: Use perturbation method
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Feature importance semantic axes test - alternative method
            {
                'name': 'tsne_importance_axes',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'semantic_axes': True,
                'semantic_axes_method': 'feature_importance',  # NEW: Use feature importance method
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Perturbation + metadata test - combine perturbation method with metadata
            {
                'name': 'tsne_perturbation_metadata',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'semantic_axes': True,
                'semantic_axes_method': 'perturbation',  # NEW: Use perturbation method
                'use_metadata': True,  # NEW: Enable metadata
                'auto_load_metadata': True,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # 3D perturbation test - test perturbation method with 3D visualization
            {
                'name': 'tsne_3d_perturbation',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'semantic_axes': True,
                'semantic_axes_method': 'perturbation',  # NEW: Use perturbation method
                'use_3d_tsne': True,  # NEW: 3D visualization
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15
            },
            # Multi-viz with perturbation method
            {
                'name': 'multi_perturbation_axes',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne'],
                'layout_strategy': 'sequential',
                'reasoning_focus': 'comparison',
                'use_semantic_names': True,
                'load_semantic_from_cc18': True,
                'nn_k': 30,
                'semantic_axes': True,
                'semantic_axes_method': 'perturbation'  # NEW: Use perturbation method
            }
        ]
        
        # Different modality parameters and mlxtend methods
        parameter_configs = [
            # High DPI visualization
            {
                'name': 'tsne_high_dpi',
                'enable_multi_viz': False,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15,
                'image_dpi': 150
            },
            # Different zoom factor
            {
                'name': 'tsne_zoomed',
                'enable_multi_viz': False,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'nn_k': 30,
                'tsne_perplexity': 15,
                'tsne_zoom_factor': 3.0
            },
            # MLxtend frequent patterns
            {
                'name': 'frequent_patterns',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'frequent_patterns'],
                'layout_strategy': 'sequential',
                'nn_k': 30,
                'reasoning_focus': 'comparison'
            },
            # MLxtend decision regions with SVM
            {
                'name': 'decision_regions_svm',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'decision_regions'],
                'layout_strategy': 'sequential',
                'reasoning_focus': 'comparison',
                'nn_k': 30,
                'decision_classifier': 'svm'
            },
            # Metadata testing with comprehensive info
            {
                'name': 'metadata_comprehensive',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne', 'isomap'],
                'layout_strategy': 'hierarchical',
                'reasoning_focus': 'comparison',
                'include_metadata': True,
                'nn_k': 30,
                'metadata_types': ['quality_metrics', 'timing_info', 'method_params']
            }
        ]
        
        # Combine all configurations - semantic tests first
        all_configs = (semantic_configs + single_viz_configs + multi_viz_configs + 
                      parameter_configs)
        
        return all_configs
    
    def run_single_test(self, config: Dict[str, Any], test_idx: int) -> Dict[str, Any]:
        """Run a single test configuration."""
        logger.info(f"Running test {test_idx + 1}: {config['name']}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Use fixed test indices for all tests
        if self.test_indices is None:
            # First test - establish the indices
            if len(X_test) > self.responses_per_test:
                self.test_indices = np.random.choice(len(X_test), self.responses_per_test, replace=False)
                self.test_indices.sort()  # Keep sorted for consistency
            else:
                self.test_indices = np.arange(len(X_test))
        
        # Use the same indices for all tests
        X_test = X_test[self.test_indices]
        y_test = y_test[self.test_indices]
        
        # Create test-specific directories
        test_dir = self.output_dir / f"test_{test_idx:02d}_{config['name']}"
        test_dir.mkdir(exist_ok=True)
        (test_dir / "responses").mkdir(exist_ok=True)
        (test_dir / "prompts").mkdir(exist_ok=True)
        
        try:
            # Load semantic class names if requested
            semantic_class_names = None
            if config.get('load_semantic_from_cc18', False):
                semantic_class_names = self._load_semantic_class_names(self.task_id, len(np.unique(y_train)))
                if semantic_class_names:
                    config['semantic_class_names'] = semantic_class_names
                    logger.info(f"Loaded semantic class names for test {test_idx + 1}: {semantic_class_names}")
                else:
                    logger.info(f"No semantic class names found for task {self.task_id}, using Class_<NUM> fallback")
                    config['semantic_class_names'] = [f"Class_{i}" for i in range(len(np.unique(y_train)))]
                    config['use_semantic_names'] = False  # Disable semantic names since we couldn't load them
            
            # Create classifier with configuration
            classifier_config = {
                'modality': 'tabular',
                'vlm_model_id': self.vlm_model,
                'tsne_perplexity': config.get('tsne_perplexity', 15),
                'tsne_n_iter': 500,  # Reduced for speed
                'seed': 42,
                'max_vlm_image_size': 1024,  # Reduced for speed
                'image_dpi': config.get('image_dpi', 100),
                'zoom_factor': config.get('zoom_factor', self.zoom_factor),  # Use instance variable as default
                'use_semantic_names': config.get('use_semantic_names', False),
                # NEW: Add semantic axes and metadata parameters
                'semantic_axes': config.get('semantic_axes', False),
                'semantic_axes_method': config.get('semantic_axes_method', 'pca_loadings'),  # NEW: Support different semantic axes methods
                'use_metadata': config.get('use_metadata', False),
                # Pass feature names for semantic axes computation
                'feature_names': self.dataset_info.get('feature_names', None),
                'auto_load_metadata': config.get('auto_load_metadata', True),
                # VLM model parameters to avoid KV cache issues
                'max_model_len': 16384,
                'gpu_memory_utilization': 0.7,  # Reduced to be safer
                'backend': self.backend  # Use backend from command line
            }
            
            # Add single or multi-viz specific parameters
            if config.get('enable_multi_viz', False):
                # Build multi_viz_config with method-specific parameters
                multi_viz_config = {}
                
                # Check if decision_regions is in the visualization methods
                if 'decision_regions' in config.get('visualization_methods', []):
                    multi_viz_config['decision_regions'] = {
                        'decision_classifier': config.get('decision_classifier', 'svm'),
                        'embedding_method': 'pca'
                    }
                
                classifier_config.update({
                    'enable_multi_viz': True,
                    'visualization_methods': config.get('visualization_methods', ['tsne']),
                    'layout_strategy': config.get('layout_strategy', 'sequential'),
                    'reasoning_focus': config.get('reasoning_focus', 'classification'),
                    'multi_viz_config': multi_viz_config
                })
            else:
                classifier_config.update({
                    'enable_multi_viz': False,
                    'use_3d': config.get('use_3d_tsne', False),  # Updated parameter name
                    'use_knn_connections': config.get('use_knn_connections', False),
                    'nn_k': config.get('knn_k', 5)  # Updated parameter name
                })
            
            classifier = ClamTsneClassifier(**classifier_config)
            
            # Fit the classifier first
            # Pass semantic class names if provided
            fit_kwargs = {
                'dataset_info': self.dataset_info  # Pass dataset info for metadata loading
            }
            if config.get('use_semantic_names', False) and config.get('semantic_class_names'):
                fit_kwargs['class_names'] = config['semantic_class_names'][:len(np.unique(y_train))]
            
            classifier.fit(X_train, y_train, X_test, **fit_kwargs)
            
            # Use CLAM's evaluate method to get detailed prediction information
            try:
                results = classifier.evaluate(
                    X_test, 
                    y_test, 
                    return_detailed=True,
                    save_outputs=True,
                    output_dir=str(test_dir),
                    visualization_save_cadence=3
                )
                
                # Extract ground truth labels
                ground_truth_labels = [int(label) for label in y_test]
                
                # Extract actual VLM responses and prompts from prediction_details
                all_responses = []
                all_prompts = []
                
                if 'prediction_details' in results and results['prediction_details']:
                    prediction_details = results['prediction_details']
                    
                    # Extract VLM responses
                    for detail in prediction_details:
                        if 'vlm_response' in detail:
                            all_responses.append(detail['vlm_response'])
                        else:
                            # Fallback if vlm_response not available
                            parsed_pred = detail.get('parsed_prediction', 'UNKNOWN')
                            all_responses.append(f"Class: Class_{parsed_pred} | Reasoning: Parsed prediction from CLAM")
                    
                    logger.info(f"Extracted {len(all_responses)} actual VLM responses from prediction_details")
                    
                    # For prompts, we'll generate a representative one since CLAM doesn't store the exact prompts
                    # in prediction_details, but we know what prompt structure was used
                    if config.get('enable_multi_viz', False):
                        # Multi-viz prompt
                        multi_viz_info = []
                        if hasattr(classifier, 'context_composer') and classifier.context_composer:
                            for viz in classifier.context_composer.visualizations:
                                multi_viz_info.append({
                                    'method': viz.method_name,
                                    'description': f"{viz.method_name} visualization"
                                })
                        
                        from clam.utils.vlm_prompting import create_classification_prompt
                        # Use semantic class names if available, otherwise generic ones
                        prompt_class_names = config.get('semantic_class_names', [f"Class_{i}" for i in range(len(np.unique(y_train)))])
                        sample_prompt = create_classification_prompt(
                            class_names=prompt_class_names,
                            modality='tabular',
                            dataset_description=f"Test dataset with {len(np.unique(y_train))} classes",
                            use_semantic_names=config.get('use_semantic_names', False),
                            multi_viz_info=multi_viz_info
                        )
                    else:
                        # Single viz prompt
                        from clam.utils.vlm_prompting import create_classification_prompt
                        # Use semantic class names if available, otherwise generic ones
                        prompt_class_names = config.get('semantic_class_names', [f"Class_{i}" for i in range(len(np.unique(y_train)))])
                        sample_prompt = create_classification_prompt(
                            class_names=prompt_class_names,
                            modality='tabular',
                            use_knn=config.get('use_knn_connections', False),
                            use_3d=config.get('use_3d_tsne', False),
                            knn_k=config.get('knn_k', 5),
                            dataset_description=f"Test dataset with {len(np.unique(y_train))} classes",
                            use_semantic_names=config.get('use_semantic_names', False)
                        )
                    
                    # Use the same prompt for all samples (this is typically how CLAM works)
                    all_prompts = [sample_prompt] * len(all_responses)
                
                else:
                    # Fallback if no prediction_details available
                    logger.warning("No prediction_details found in results")
                    predictions = results.get('predictions', [])
                    if isinstance(predictions, (list, np.ndarray)):
                        all_responses = [f"Class: Class_{pred} | Reasoning: CLAM prediction (no detailed response)" for pred in predictions]
                    else:
                        all_responses = [f"Class: UNKNOWN | Reasoning: No prediction details available" for _ in range(len(X_test))]
                    
                    # Generate fallback prompts
                    # Generate appropriate class names for fallback
                    if config.get('use_semantic_names', False) and config.get('semantic_class_names'):
                        fallback_class_names = config['semantic_class_names'][:len(np.unique(y_train))]
                    else:
                        fallback_class_names = [f"Class_{i}" for i in range(len(np.unique(y_train)))]
                    
                    from clam.utils.vlm_prompting import create_classification_prompt
                    sample_prompt = create_classification_prompt(
                        class_names=fallback_class_names,
                        modality='tabular',
                        dataset_description=f"Test dataset with {len(np.unique(y_train))} classes",
                        use_semantic_names=config.get('use_semantic_names', False)
                    )
                    all_prompts = [sample_prompt] * len(all_responses)
                
                # Save the actual outputs manually since CLAM's save_outputs isn't implemented
                # Save detailed VLM information to the test directory
                if 'prediction_details' in results and results['prediction_details']:
                    detailed_output = {
                        'test_config': config,
                        'ground_truth': ground_truth_labels,
                        'prediction_details': results['prediction_details'],
                        'completion_rate': results.get('completion_rate', 1.0),
                        'test_indices': self.test_indices.tolist() if self.test_indices is not None else []
                    }
                    
                    detailed_output_path = test_dir / "detailed_vlm_outputs.json"
                    with open(detailed_output_path, 'w') as f:
                        json.dump(detailed_output, f, indent=2, default=str)
                    logger.info(f"Saved detailed VLM outputs to {detailed_output_path}")
                    
                    # Calculate and log metrics using unified metrics system
                    metrics = self._calculate_metrics(
                        results['prediction_details'], 
                        ground_truth_labels, 
                        config['name']
                    )
                    
                    if metrics:
                        # Log metrics using unified metrics logger
                        metrics_logger = MetricsLogger(
                            model_name=f"CLAM-{config['name']}",
                            dataset_name=self.dataset_info.get('name', f'dataset_{self.task_id}'),
                            use_wandb=False,  # Disable W&B for test script
                            logger=logger
                        )
                        
                        # Log all calculated metrics
                        metrics_logger.log_all_metrics(metrics)
                        
                        # Add metrics to detailed output
                        detailed_output['metrics'] = metrics
                        
                        # Save updated detailed output with metrics
                        with open(detailed_output_path, 'w') as f:
                            json.dump(detailed_output, f, indent=2, default=str)
                        
                        logger.info(f"✓ Test {test_idx + 1} metrics: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_macro']:.3f}")
                    else:
                        logger.warning(f"Could not calculate metrics for test {test_idx + 1}")
                        metrics = {'error': 'Could not calculate metrics'}
                
                # Save individual response and prompt files in test directory
                for i, (response, prompt) in enumerate(zip(all_responses, all_prompts)):
                    response_file = test_dir / "responses" / f"response_{i:02d}.txt"
                    with open(response_file, 'w') as f:
                        f.write(response)
                    
                    prompt_file = test_dir / "prompts" / f"prompt_{i:02d}.txt"
                    with open(prompt_file, 'w') as f:
                        f.write(prompt)
                
                # Find and copy CLAM's generated visualizations
                viz_path = self.output_dir / "visualizations" / f"test_{test_idx:02d}_{config['name']}.png"
                
                # CLAM saves visualizations in its temp directory during prediction
                viz_found = False
                if hasattr(classifier, 'temp_dir') and classifier.temp_dir and os.path.exists(classifier.temp_dir):
                    temp_viz_files = list(Path(classifier.temp_dir).glob("*.png"))
                    if temp_viz_files:
                        # Copy the most recent visualization file
                        latest_viz = max(temp_viz_files, key=lambda p: p.stat().st_mtime)
                        shutil.copy2(latest_viz, viz_path)
                        logger.info(f"Copied CLAM visualization from {latest_viz} to {viz_path}")
                        viz_found = True
                        
                        # Also copy any additional visualizations for multi-viz tests
                        if len(temp_viz_files) > 1:
                            for i, viz_file in enumerate(temp_viz_files):
                                additional_viz_path = test_dir / f"visualization_{i}.png"
                                shutil.copy2(viz_file, additional_viz_path)
                
                if not viz_found:
                    logger.warning(f"No visualization found for test {test_idx + 1}. Check CLAM's temp_dir: {getattr(classifier, 'temp_dir', 'Not set')}")
                
                logger.info(f"✓ Test {test_idx + 1} completed successfully using real CLAM pipeline")
                
            except Exception as e:
                logger.error(f"✗ CLAM pipeline failed for test {test_idx + 1}: {e}")
                # Create fallback data for summary
                ground_truth_labels = [int(label) for label in y_test]
                all_responses = [f"Class: FAILED | Reasoning: Pipeline error: {str(e)}"] * len(X_test)
                all_prompts = ["FAILED"] * len(X_test)
                viz_path = self.output_dir / "visualizations" / f"test_{test_idx:02d}_{config['name']}_failed.png"
            
            # Save aggregated prompt and responses
            prompt_path = self.output_dir / "prompts" / f"test_{test_idx:02d}_{config['name']}_all.txt"
            with open(prompt_path, 'w') as f:
                f.write("\n\n=== PROMPT ===\n\n".join(all_prompts))
            
            response_path = self.output_dir / "responses" / f"test_{test_idx:02d}_{config['name']}_all.txt"
            with open(response_path, 'w') as f:
                f.write("\n\n=== RESPONSE ===\n\n".join(all_responses))
            
            # Save ground truth
            # Use semantic class names if provided
            if config.get('use_semantic_names', False) and config.get('semantic_class_names'):
                gt_class_names = config['semantic_class_names'][:len(np.unique(y_train))]
            else:
                gt_class_names = [f"Class_{i}" for i in range(len(np.unique(y_train)))]
            
            ground_truth_path = self.output_dir / "ground_truth" / f"test_{test_idx:02d}_{config['name']}.json"
            ground_truth_data = {
                'test_indices': self.test_indices.tolist() if self.test_indices is not None else [],
                'ground_truth_labels': ground_truth_labels,
                'class_names': gt_class_names
            }
            
            # Add metadata information if this is a metadata test
            if config.get('include_metadata', False):
                ground_truth_data['metadata_config'] = {
                    'metadata_types': config.get('metadata_types', []),
                    'visualization_methods': config.get('visualization_methods', []),
                    'note': 'This test includes comprehensive metadata from visualization methods'
                }
            
            with open(ground_truth_path, 'w') as f:
                json.dump(ground_truth_data, f, indent=2)
            
            # Save configuration
            config_path = self.output_dir / "configs" / f"test_{test_idx:02d}_{config['name']}.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Collect test result
            result = {
                'test_idx': test_idx,
                'config_name': config['name'],
                'success': True,
                'num_test_samples': len(X_test),
                'test_indices': self.test_indices.tolist() if self.test_indices is not None else [],
                'ground_truth_labels': ground_truth_labels,
                'avg_prompt_length': np.mean([len(p) for p in all_prompts]),
                'visualization_path': str(viz_path.relative_to(self.output_dir)),
                'prompt_path': str(prompt_path.relative_to(self.output_dir)),
                'response_path': str(response_path.relative_to(self.output_dir)),
                'ground_truth_path': str(ground_truth_path.relative_to(self.output_dir)),
                'config_path': str(config_path.relative_to(self.output_dir)),
                'test_directory': str(test_dir.relative_to(self.output_dir)),
                'is_multi_viz': config.get('enable_multi_viz', False),
                'visualization_methods': config.get('visualization_methods', ['tsne']),
                'metrics': metrics if 'metrics' in locals() else None,
                'error': None
            }
            
            logger.info(f"✓ Test {test_idx + 1} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"✗ Test {test_idx + 1} failed: {e}")
            result = {
                'test_idx': test_idx,
                'config_name': config['name'],
                'success': False,
                'error': str(e)
            }
            return result
    
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test configurations."""
        logger.info("Starting comprehensive VLM prompting test suite")
        logger.info(f"Dataset: {self.dataset_info['name']} (OpenML ID: {self.task_id})")
        logger.info(f"VLM Model: {self.vlm_model}")
        logger.info(f"Output Directory: {self.output_dir}")
        
        configs = self._get_test_configurations()
        
        # Handle specific target config selection
        if hasattr(self, '_target_config') and self._target_config:
            matching_configs = [config for config in configs if config['name'] == self._target_config]
            if matching_configs:
                configs = [matching_configs[0]]
                logger.info(f"Running specific test configuration: {self._target_config}")
            else:
                logger.error(f"Target configuration '{self._target_config}' not found")
                return {}
        else:
            # Limit the number of test configurations if specified
            if self.num_tests is not None and self.num_tests < len(configs):
                configs = configs[:self.num_tests]
                logger.info(f"Limited to {self.num_tests} test configurations (out of {len(self._get_test_configurations())} available)")
        
        logger.info(f"Running {len(configs)} test configurations with {self.num_samples_per_test} samples each...")
        
        for i, config in enumerate(configs):
            result = self.run_single_test(config, i)
            self.test_results.append(result)
        
        # Generate summary
        summary = self._generate_summary()
        
        # Save summary using unified JSON utilities
        summary_path = self.output_dir / "test_summary.json"
        safe_json_dump(summary, str(summary_path), logger=logger, indent=2)
        
        # Save detailed results using unified JSON utilities
        results_path = self.output_dir / "detailed_results.json"
        safe_json_dump(self.test_results, str(results_path), logger=logger, indent=2)
        
        logger.info(f"Test suite completed. Results saved to {self.output_dir}")
        return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary with comparative analysis."""
        successful_tests = [r for r in self.test_results if r.get('success', False)]
        failed_tests = [r for r in self.test_results if not r.get('success', False)]
        
        single_viz_tests = [r for r in successful_tests if not r.get('is_multi_viz', False)]
        multi_viz_tests = [r for r in successful_tests if r.get('is_multi_viz', False)]
        
        # Count visualization methods used
        method_counts = {}
        for result in multi_viz_tests:
            methods = result.get('visualization_methods', [])
            for method in methods:
                method_counts[method] = method_counts.get(method, 0) + 1
        
        # Aggregate metrics for comparative analysis
        config_metrics = {}
        overall_metrics = {'accuracy': [], 'f1_macro': [], 'balanced_accuracy': []}
        
        for result in successful_tests:
            if result.get('metrics'):
                config_name = result['config_name']
                metrics = result['metrics']
                
                config_metrics[config_name] = metrics
                
                # Collect for overall statistics
                if metrics.get('accuracy') is not None:
                    overall_metrics['accuracy'].append(metrics['accuracy'])
                if metrics.get('f1_macro') is not None:
                    overall_metrics['f1_macro'].append(metrics['f1_macro'])
                if metrics.get('balanced_accuracy') is not None:
                    overall_metrics['balanced_accuracy'].append(metrics['balanced_accuracy'])
        
        # Calculate aggregate statistics
        aggregate_stats = {}
        for metric_name, values in overall_metrics.items():
            if values:
                aggregate_stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        # Find best performing configurations
        best_configs = {}
        for metric_name in ['accuracy', 'f1_macro', 'balanced_accuracy']:
            best_score = -1
            best_config = None
            for config_name, metrics in config_metrics.items():
                if metrics.get(metric_name, -1) > best_score:
                    best_score = metrics[metric_name]
                    best_config = config_name
            if best_config:
                best_configs[metric_name] = {
                    'config': best_config,
                    'score': best_score
                }
        
        # Compare single-viz vs multi-viz performance
        single_viz_metrics = [r['metrics'] for r in single_viz_tests if r.get('metrics')]
        multi_viz_metrics = [r['metrics'] for r in multi_viz_tests if r.get('metrics')]
        
        comparison = {}
        if single_viz_metrics and multi_viz_metrics:
            for metric_name in ['accuracy', 'f1_macro', 'balanced_accuracy']:
                single_scores = [m[metric_name] for m in single_viz_metrics if m.get(metric_name) is not None]
                multi_scores = [m[metric_name] for m in multi_viz_metrics if m.get(metric_name) is not None]
                
                if single_scores and multi_scores:
                    comparison[metric_name] = {
                        'single_viz_mean': np.mean(single_scores),
                        'multi_viz_mean': np.mean(multi_scores),
                        'difference': np.mean(multi_scores) - np.mean(single_scores),
                        'single_viz_count': len(single_scores),
                        'multi_viz_count': len(multi_scores)
                    }
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': self.dataset_info,
            'vlm_model': self.vlm_model,
            'total_tests': len(self.test_results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'success_rate': len(successful_tests) / len(self.test_results) * 100,
            'single_viz_tests': len(single_viz_tests),
            'multi_viz_tests': len(multi_viz_tests),
            'visualization_method_counts': method_counts,
            'average_prompt_length': np.mean([r.get('avg_prompt_length', 0) for r in successful_tests]),
            'total_test_samples': sum([r.get('num_test_samples', 0) for r in successful_tests]),
            'output_directory': str(self.output_dir),
            
            # New metrics analysis
            'overall_metrics': aggregate_stats,
            'best_performing_configs': best_configs,
            'single_vs_multi_viz_comparison': comparison,
            'config_performance': config_metrics,
            
            # Analysis insights
            'insights': {
                'best_overall_config': best_configs.get('accuracy', {}).get('config'),
                'multi_viz_advantage': {
                    metric: comp.get('difference', 0) > 0 
                    for metric, comp in comparison.items()
                } if comparison else {},
                'most_tested_viz_method': max(method_counts, key=method_counts.get) if method_counts else None,
                'configs_with_metrics': len(config_metrics)
            },
            
            'files_generated': {
                'visualizations': len(list((self.output_dir / "visualizations").glob("*.png"))),
                'prompts': len(list((self.output_dir / "prompts").glob("*.txt"))),
                'responses': len(list((self.output_dir / "responses").glob("*.txt"))),
                'configs': len(list((self.output_dir / "configs").glob("*.json"))),
                'ground_truth': len(list((self.output_dir / "ground_truth").glob("*.json"))),
                'test_directories': len([d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith('test_')])
            }
        }
        
        return summary


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Comprehensive VLM Prompting Test Suite")
    parser.add_argument("--output_dir", type=str, default="./test_vlm_outputs",
                       help="Directory to save test outputs")
    parser.add_argument("--task_id", type=int, default=23,
                       help="OpenML task ID (default: 23 = cmc, alternatives: 61=iris, 1046=wine, 1461=banknote)")
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="VLM model to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--num_tests", type=int, default=None,
                       help="Number of test configurations to run (default: None, runs all available)")
    parser.add_argument("--num_samples_per_test", type=int, default=10,
                       help="Number of test samples to process per configuration (default: 10)")
    parser.add_argument("--backend", type=str, default="auto",
                       choices=["auto", "vllm", "transformers"],
                       help="Backend to use for VLM inference (default: auto)")
    parser.add_argument("--zoom_factor", type=float, default=6.5,
                       help="Zoom factor for t-SNE visualizations (default: 6.5)")
    parser.add_argument("--test_config", type=str, default=None,
                       help="Run only a specific test configuration by name (e.g., 'tsne_perturbation_axes')")
    parser.add_argument("--list_configs", action="store_true",
                       help="List all available test configuration names and exit")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create test suite
    test_suite = VLMPromptingTestSuite(
        output_dir=args.output_dir,
        task_id=args.task_id,
        vlm_model=args.vlm_model,
        num_tests=args.num_tests,
        num_samples_per_test=args.num_samples_per_test,
        backend=args.backend,
        zoom_factor=args.zoom_factor
    )
    
    # Handle list_configs option
    if args.list_configs:
        test_configs = test_suite._get_test_configurations()
        print("\nAvailable test configurations:")
        print("=" * 50)
        for i, config in enumerate(test_configs):
            semantic_method = config.get('semantic_axes_method', 'pca_loadings')
            semantic_axes = config.get('semantic_axes', False)
            multi_viz = config.get('enable_multi_viz', False)
            metadata = config.get('use_metadata', False)
            
            status = []
            if semantic_axes:
                status.append(f"semantic_axes({semantic_method})")
            if metadata:
                status.append("metadata")
            if multi_viz:
                status.append("multi_viz")
            
            status_str = f" [{', '.join(status)}]" if status else ""
            print(f"{i+1:2d}. {config['name']}{status_str}")
        
        print(f"\nTotal: {len(test_configs)} configurations")
        print("\nUse --test_config <name> to run a specific configuration")
        return
    
    # Handle specific test config option
    if args.test_config:
        test_configs = test_suite._get_test_configurations()
        matching_configs = [config for config in test_configs if config['name'] == args.test_config]
        
        if not matching_configs:
            print(f"Error: Test configuration '{args.test_config}' not found.")
            print("Use --list_configs to see available configurations.")
            return 1
        
        print(f"Running specific test configuration: {args.test_config}")
        # Override num_tests to run only the specific config
        test_suite.num_tests = 1
        test_suite._target_config = args.test_config
    
    summary = test_suite.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("VLM PROMPTING TEST SUITE SUMMARY")
    print("=" * 60)
    print(f"Dataset: {summary['dataset_info']['name']}")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful: {summary['successful_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Single-viz Tests: {summary['single_viz_tests']}")
    print(f"Multi-viz Tests: {summary['multi_viz_tests']}")
    print(f"Total Test Samples: {summary['total_test_samples']}")
    print(f"Average Prompt Length: {summary['average_prompt_length']:.0f} characters")
    print(f"\nFiles Generated:")
    for file_type, count in summary['files_generated'].items():
        print(f"  {file_type}: {count}")
    print(f"\nOutput Directory: {summary['output_directory']}")
    print(f"Summary saved to: {args.output_dir}/test_summary.json")
    print(f"Detailed results saved to: {args.output_dir}/detailed_results.json")
    
    # Print metrics analysis
    if summary.get('overall_metrics'):
        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS ANALYSIS")
        print("=" * 60)
        
        for metric_name, stats in summary['overall_metrics'].items():
            print(f"\n{metric_name.upper()}:")
            print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  Configs tested: {stats['count']}")
        
        # Best performing configs
        if summary.get('best_performing_configs'):
            print("\nBEST PERFORMING CONFIGURATIONS:")
            for metric, best in summary['best_performing_configs'].items():
                print(f"  {metric}: {best['config']} (score: {best['score']:.3f})")
        
        # Single vs Multi-viz comparison
        if summary.get('single_vs_multi_viz_comparison'):
            print("\nSINGLE-VIZ vs MULTI-VIZ COMPARISON:")
            for metric, comp in summary['single_vs_multi_viz_comparison'].items():
                advantage = "Multi-viz" if comp['difference'] > 0 else "Single-viz"
                print(f"  {metric}: Single={comp['single_viz_mean']:.3f}, Multi={comp['multi_viz_mean']:.3f}")
                print(f"    Advantage: {advantage} (+{abs(comp['difference']):.3f})")
        
        # Key insights
        if summary.get('insights'):
            insights = summary['insights']
            print("\nKEY INSIGHTS:")
            if insights.get('best_overall_config'):
                print(f"  • Best overall configuration: {insights['best_overall_config']}")
            if insights.get('most_tested_viz_method'):
                print(f"  • Most tested visualization method: {insights['most_tested_viz_method']}")
            if insights.get('multi_viz_advantage'):
                multi_advantages = [k for k, v in insights['multi_viz_advantage'].items() if v]
                if multi_advantages:
                    print(f"  • Multi-viz shows advantage in: {', '.join(multi_advantages)}")
    
    if summary['visualization_method_counts']:
        print(f"\nVisualization Methods Used:")
        for method, count in summary['visualization_method_counts'].items():
            print(f"  {method}: {count} tests")
    
    print("\n🎉 Test suite completed successfully!")
    print("\nTo inspect results:")
    print(f"  ls {args.output_dir}/")
    print(f"  cat {args.output_dir}/test_summary.json")
    print(f"  cat {args.output_dir}/detailed_results.json")


if __name__ == "__main__":
    main()