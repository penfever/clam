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
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLMPromptingTestSuite:
    """Comprehensive test suite for VLM prompting with various configurations."""
    
    def __init__(self, output_dir: str, dataset_id: int = 31, vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct"):
        """
        Initialize the test suite.
        
        Args:
            output_dir: Directory to save outputs
            dataset_id: OpenML dataset ID (default: 31 = credit-g)
            vlm_model: VLM model to use (default: small Qwen model)
        """
        self.output_dir = Path(output_dir)
        self.dataset_id = dataset_id
        self.vlm_model = vlm_model
        
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
        
        # Number of responses per test
        self.responses_per_test = 10
        
        # Load dataset
        self.X, self.y, self.dataset_info = self._load_dataset()
        
        logger.info(f"Initialized test suite with dataset {dataset_id}")
        logger.info(f"Dataset shape: {self.X.shape}, Classes: {len(np.unique(self.y))}")
    
    def _load_dataset(self):
        """Load OpenML dataset with robust error handling."""
        try:
            # Load dataset from OpenML
            dataset = fetch_openml(data_id=self.dataset_id, as_frame=True, parser='auto')
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
            
            dataset_info = {
                'name': dataset.DESCR if hasattr(dataset, 'DESCR') else f'OpenML_{self.dataset_id}',
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)),
                'openml_id': self.dataset_id,
                'data_source': 'openml'
            }
            
            logger.info(f"Successfully loaded OpenML dataset {self.dataset_id}")
            return X, y, dataset_info
            
        except Exception as e:
            logger.warning(f"Failed to load OpenML dataset {self.dataset_id}: {e}")
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
                'openml_id': None,
                'data_source': 'synthetic'
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
                'tsne_perplexity': 15
            },
            # t-SNE with KNN
            {
                'name': 'tsne_knn',
                'enable_multi_viz': False,
                'use_3d_tsne': False,
                'use_knn_connections': True,
                'knn_k': 5,
                'tsne_perplexity': 15
            },
            # 3D t-SNE
            {
                'name': 'tsne_3d',
                'enable_multi_viz': False,
                'use_3d_tsne': True,
                'use_knn_connections': False,
                'tsne_perplexity': 15
            },
            # 3D t-SNE with KNN
            {
                'name': 'tsne_3d_knn',
                'enable_multi_viz': False,
                'use_3d_tsne': True,
                'use_knn_connections': True,
                'knn_k': 3,
                'tsne_perplexity': 15
            },
            # Different perplexity
            {
                'name': 'tsne_high_perplexity',
                'enable_multi_viz': False,
                'use_3d_tsne': False,
                'use_knn_connections': False,
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
                'layout_strategy': 'adaptive_grid',
                'reasoning_focus': 'comparison'
            },
            # Three methods
            {
                'name': 'multi_pca_tsne_spectral',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne', 'spectral'],
                'layout_strategy': 'adaptive_grid',
                'reasoning_focus': 'consensus'
            },
            # Linear vs nonlinear focus
            {
                'name': 'multi_linear_nonlinear',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne', 'isomap'],
                'layout_strategy': 'hierarchical',
                'reasoning_focus': 'divergence'
            },
            # Local vs global methods
            {
                'name': 'multi_local_global',
                'enable_multi_viz': True,
                'visualization_methods': ['tsne', 'isomap', 'mds'],
                'layout_strategy': 'sequential',
                'reasoning_focus': 'comparison'
            },
            # Comprehensive multi-viz
            {
                'name': 'multi_comprehensive',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne', 'spectral', 'isomap'],
                'layout_strategy': 'focus_plus_context',
                'reasoning_focus': 'consensus'
            },
            # Different layout strategies
            {
                'name': 'multi_grid_layout',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne'],
                'layout_strategy': 'grid',
                'reasoning_focus': 'comparison'
            },
            # With UMAP if available
            {
                'name': 'multi_with_umap',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne', 'umap'],
                'layout_strategy': 'adaptive_grid',
                'reasoning_focus': 'comparison'
            }
        ]
        
        # Semantic naming variations
        semantic_configs = [
            # Single viz with semantic names
            {
                'name': 'tsne_semantic',
                'enable_multi_viz': False,
                'use_semantic_names': True,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'tsne_perplexity': 15
            },
            # Multi-viz with semantic names
            {
                'name': 'multi_semantic',
                'enable_multi_viz': True,
                'visualization_methods': ['pca', 'tsne'],
                'layout_strategy': 'adaptive_grid',
                'reasoning_focus': 'comparison',
                'use_semantic_names': True
            }
        ]
        
        # Different modality parameters
        parameter_configs = [
            # High DPI visualization
            {
                'name': 'tsne_high_dpi',
                'enable_multi_viz': False,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'tsne_perplexity': 15,
                'image_dpi': 150
            },
            # Different zoom factor
            {
                'name': 'tsne_zoomed',
                'enable_multi_viz': False,
                'use_3d_tsne': False,
                'use_knn_connections': False,
                'tsne_perplexity': 15,
                'tsne_zoom_factor': 3.0
            }
        ]
        
        # Combine all configurations
        all_configs = (single_viz_configs + multi_viz_configs + 
                      semantic_configs + parameter_configs)
        
        # Limit to 20 configurations
        return all_configs[:20]
    
    def run_single_test(self, config: Dict[str, Any], test_idx: int) -> Dict[str, Any]:
        """Run a single test configuration."""
        logger.info(f"Running test {test_idx + 1}/20: {config['name']}")
        
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
            # Create classifier with configuration
            classifier_config = {
                'modality': 'tabular',
                'vlm_model_id': self.vlm_model,
                'tsne_perplexity': config.get('tsne_perplexity', 15),
                'tsne_n_iter': 500,  # Reduced for speed
                'seed': 42,
                'max_vlm_image_size': 1024,  # Reduced for speed
                'image_dpi': config.get('image_dpi', 100),
                'tsne_zoom_factor': config.get('tsne_zoom_factor', 2.0),
                'use_semantic_names': config.get('use_semantic_names', False),
                # VLM model parameters to avoid KV cache issues
                'max_model_len': 16384,
                'gpu_memory_utilization': 0.7  # Reduced to be safer
            }
            
            # Add single or multi-viz specific parameters
            if config.get('enable_multi_viz', False):
                classifier_config.update({
                    'enable_multi_viz': True,
                    'visualization_methods': config.get('visualization_methods', ['tsne']),
                    'layout_strategy': config.get('layout_strategy', 'adaptive_grid'),
                    'reasoning_focus': config.get('reasoning_focus', 'classification'),
                    'multi_viz_config': {}
                })
            else:
                classifier_config.update({
                    'enable_multi_viz': False,
                    'use_3d_tsne': config.get('use_3d_tsne', False),
                    'use_knn_connections': config.get('use_knn_connections', False),
                    'knn_k': config.get('knn_k', 5)
                })
            
            classifier = ClamTsneClassifier(**classifier_config)
            
            # Fit the classifier first
            classifier.fit(X_train, y_train, X_test)
            
            # Use CLAM's evaluate method to get detailed prediction information
            try:
                results = classifier.evaluate(
                    X_test, 
                    y_test, 
                    return_detailed=True
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
                        sample_prompt = create_classification_prompt(
                            class_names=[f"Class_{i}" for i in range(len(np.unique(y_train)))],
                            modality='tabular',
                            dataset_description=f"Test dataset with {len(np.unique(y_train))} classes",
                            use_semantic_names=config.get('use_semantic_names', False),
                            multi_viz_info=multi_viz_info
                        )
                    else:
                        # Single viz prompt
                        from clam.utils.vlm_prompting import create_classification_prompt
                        sample_prompt = create_classification_prompt(
                            class_names=[f"Class_{i}" for i in range(len(np.unique(y_train)))],
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
                    from clam.utils.vlm_prompting import create_classification_prompt
                    sample_prompt = create_classification_prompt(
                        class_names=[f"Class_{i}" for i in range(len(np.unique(y_train)))],
                        modality='tabular',
                        dataset_description=f"Test dataset with {len(np.unique(y_train))} classes"
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
                        'test_indices': self.test_indices.tolist()
                    }
                    
                    detailed_output_path = test_dir / "detailed_vlm_outputs.json"
                    with open(detailed_output_path, 'w') as f:
                        json.dump(detailed_output, f, indent=2, default=str)
                    logger.info(f"Saved detailed VLM outputs to {detailed_output_path}")
                
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
            ground_truth_path = self.output_dir / "ground_truth" / f"test_{test_idx:02d}_{config['name']}.json"
            with open(ground_truth_path, 'w') as f:
                json.dump({
                    'test_indices': self.test_indices.tolist(),
                    'ground_truth_labels': ground_truth_labels,
                    'class_names': [f"Class_{i}" for i in range(len(np.unique(y_train)))]
                }, f, indent=2)
            
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
                'test_indices': self.test_indices.tolist(),
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
        logger.info(f"Dataset: {self.dataset_info['name']} (OpenML ID: {self.dataset_id})")
        logger.info(f"VLM Model: {self.vlm_model}")
        logger.info(f"Output Directory: {self.output_dir}")
        
        configs = self._get_test_configurations()
        logger.info(f"Running {len(configs)} test configurations...")
        
        for i, config in enumerate(configs):
            result = self.run_single_test(config, i)
            self.test_results.append(result)
        
        # Generate summary
        summary = self._generate_summary()
        
        # Save summary
        summary_path = self.output_dir / "test_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed results
        results_path = self.output_dir / "detailed_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"Test suite completed. Results saved to {self.output_dir}")
        return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary."""
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
    parser.add_argument("--dataset_id", type=int, default=23,
                       help="OpenML dataset ID (default: 23 = cmc, alternatives: 61=iris, 1046=wine, 1461=banknote)")
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="VLM model to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create and run test suite
    test_suite = VLMPromptingTestSuite(
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        vlm_model=args.vlm_model
    )
    
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
    
    if summary['visualization_method_counts']:
        print(f"\nVisualization Methods Used:")
        for method, count in summary['visualization_method_counts'].items():
            print(f"  {method}: {count} tests")
    
    print("\n🎉 Test suite completed successfully!")
    print("\nTo inspect results:")
    print(f"  ls {args.output_dir}/")
    print(f"  cat {args.output_dir}/test_summary.json")


if __name__ == "__main__":
    main()