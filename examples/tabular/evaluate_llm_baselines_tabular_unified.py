#!/usr/bin/env python
"""
EXAMPLE: Updated LLM Baselines Evaluation with Unified Results Management

This is an example showing how to update an existing evaluation script 
to use the new unified results management system while maintaining 
backward compatibility.

Key changes demonstrated:
1. Import the new results manager
2. Add option to use unified results storage
3. Convert evaluation results to standardized format
4. Save results with comprehensive metadata
5. Maintain backward compatibility with existing scripts

Usage examples:
    # Use new unified results manager
    python evaluate_llm_baselines_tabular_unified.py --dataset_name adult --use_unified_results
    
    # Use legacy results format (backward compatibility)
    python evaluate_llm_baselines_tabular_unified.py --dataset_name adult --output_dir ./legacy_results
    
    # Migrate existing results to new format
    python evaluate_llm_baselines_tabular_unified.py --migrate_from ./legacy_results
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
import glob
import datetime
import random
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union

# Add project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from clam.data import load_datasets
from sklearn.model_selection import train_test_split
from clam.utils import setup_logging, timeout_context, MetricsLogger

# NEW: Import unified results management
from clam.utils import (
    get_results_manager,
    ExperimentMetadata,
    EvaluationResults,
    ResultsArtifacts,
    save_results_unified,
    migrate_legacy_results
)

# Import legacy save function for backward compatibility
from clam.utils import save_results

# Import wandb conditionally to avoid dependency issues if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import GPU monitoring and JSON utilities
from clam.utils import (
    init_wandb_with_gpu_monitoring, 
    cleanup_gpu_monitoring, 
    GPUMonitor,
    safe_json_dump, 
    convert_for_json_serialization
)

# Import centralized argument parser
from clam.utils.evaluation_args import create_tabular_llm_evaluation_parser

# Import metadata validation utilities
from clam.utils.metadata_validation import validate_metadata_for_models

# Import LLM baseline evaluation functions  
from examples.tabular.llm_baselines.tabllm_baseline import evaluate_tabllm
from examples.tabular.llm_baselines.tabula_8b_baseline import evaluate_tabula_8b
from examples.tabular.llm_baselines.jolt_baseline import evaluate_jolt
from clam.models.clam_tsne import evaluate_clam_tsne


def convert_to_standardized_results(
    result: Dict[str, Any], 
    model_name: str, 
    dataset: Dict[str, Any],
    args: Any
) -> Tuple[EvaluationResults, ExperimentMetadata, Optional[ResultsArtifacts]]:
    """
    Convert legacy result format to standardized format.
    
    This function shows how to extract and convert existing result dictionaries
    to the new standardized format.
    """
    
    # Extract core metrics into EvaluationResults
    evaluation_results = EvaluationResults(
        accuracy=result.get('accuracy'),
        balanced_accuracy=result.get('balanced_accuracy'),
        r2_score=result.get('r2_score'),
        mae=result.get('mae'),
        rmse=result.get('rmse'),
        precision_macro=result.get('precision_macro'),
        recall_macro=result.get('recall_macro'),
        f1_macro=result.get('f1_macro'),
        precision_weighted=result.get('precision_weighted'),
        recall_weighted=result.get('recall_weighted'),
        f1_weighted=result.get('f1_weighted'),
        classification_report=result.get('classification_report'),
        confusion_matrix=result.get('confusion_matrix'),
        completion_rate=result.get('completion_rate'),
        total_prediction_time=result.get('prediction_time'),
        predictions=result.get('predictions'),
        true_labels=result.get('true_labels'),
        raw_responses=result.get('raw_responses'),
        status="completed" if not result.get('error') else "failed",
        error_message=result.get('error')
    )
    
    # Create comprehensive experiment metadata
    experiment_metadata = ExperimentMetadata(
        model_name=model_name,
        dataset_id=str(dataset.get('id', dataset.get('name', 'unknown'))),
        modality="tabular",
        
        # Dataset information
        num_samples_train=len(dataset.get('X_train', dataset.get('X', []))),
        num_samples_test=result.get('num_test_samples'),
        num_features=getattr(dataset.get('X_train', dataset.get('X', [])), 'shape', [None, None])[1],
        num_classes=result.get('num_classes'),
        class_names=result.get('class_names'),
        task_type="classification" if result.get('accuracy') is not None else "regression",
        
        # Model configuration
        model_config={
            'model_name': model_name,
            'backend': getattr(args, 'backend', None),
            'max_context_length': getattr(args, 'max_context_length', None),
            'temperature': getattr(args, 'temperature', None) if hasattr(args, 'temperature') else None,
        },
        hyperparameters={
            'k_shot': getattr(args, 'k_shot', None),
            'num_few_shot_examples': getattr(args, 'num_few_shot_examples', None),
            'balanced_few_shot': getattr(args, 'balanced_few_shot', False),
            'feature_selection_threshold': getattr(args, 'feature_selection_threshold', None),
        },
        
        # Experiment setup
        random_seed=getattr(args, 'seed', None),
        k_shot=getattr(args, 'k_shot', None),
        use_wandb=getattr(args, 'use_wandb', False),
        
        # Computational resources
        device=getattr(args, 'device', None),
        training_time_seconds=result.get('training_time'),
        evaluation_time_seconds=result.get('prediction_time'),
        
        # Additional context
        notes=f"Evaluated using {model_name} baseline on {dataset.get('name', 'unknown')} dataset",
        tags=[model_name, "tabular", "llm_baseline"]
    )
    
    # Create artifacts if available
    artifacts = None
    if model_name == 'clam_tsne' and result.get('visualizations_saved'):
        artifacts = ResultsArtifacts(
            visualizations={"tsne_plot": "tsne_visualization.png"},
            raw_outputs="vlm_responses.json" if result.get('raw_responses') else None
        )
    
    return evaluation_results, experiment_metadata, artifacts


def save_results_unified_format(
    results: List[Dict[str, Any]], 
    dataset: Dict[str, Any],
    args: Any,
    logger: logging.Logger
):
    """
    Save results using the new unified format.
    
    This function demonstrates how to integrate the new results manager
    into existing evaluation workflows.
    """
    
    results_manager = get_results_manager()
    
    for result in results:
        model_name = result.get('model_name', 'unknown_model')
        
        try:
            # Convert to standardized format
            eval_results, metadata, artifacts = convert_to_standardized_results(
                result, model_name, dataset, args
            )
            
            # Save using unified results manager
            experiment_dir = results_manager.save_evaluation_results(
                model_name=model_name,
                dataset_id=metadata.dataset_id,
                modality="tabular",
                results=eval_results,
                experiment_metadata=metadata,
                artifacts=artifacts
            )
            
            logger.info(f"Saved {model_name} results to unified format: {experiment_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save {model_name} results in unified format: {e}")
            # Fall back to legacy format
            logger.info("Falling back to legacy save format")
            save_results([result], args.output_dir, dataset['name'])


def parse_args():
    """Parse command line arguments with new unified results options."""
    parser = create_tabular_llm_evaluation_parser("Evaluate LLM baselines with unified results management")
    
    # Add unified results management options
    results_group = parser.add_argument_group('Results Management Options')
    results_group.add_argument(
        "--use_unified_results",
        action="store_true",
        help="Use the new unified results management system"
    )
    results_group.add_argument(
        "--migrate_from",
        type=str,
        help="Migrate legacy results from specified directory"
    )
    results_group.add_argument(
        "--list_experiments",
        action="store_true",
        help="List existing experiments and exit"
    )
    results_group.add_argument(
        "--generate_report",
        type=str,
        help="Generate summary report and save to specified file"
    )
    
    # Set defaults
    parser.set_defaults(
        output_dir="./llm_baseline_results"
    )
    
    args = parser.parse_args()
    
    return args


def handle_special_commands(args, logger):
    """Handle special commands like migration, listing, reporting."""
    
    if args.migrate_from:
        logger.info(f"Migrating legacy results from: {args.migrate_from}")
        stats = migrate_legacy_results(args.migrate_from, dry_run=False)
        logger.info(f"Migration completed: {stats['successful']}/{stats['total_files']} successful")
        return True
    
    if args.list_experiments:
        results_manager = get_results_manager()
        experiments = results_manager.list_experiments(modality="tabular")
        
        print(f"Found {len(experiments)} tabular experiments:")
        print(f"{'Dataset':<20} {'Model':<20} {'Path'}")
        print("-" * 70)
        
        for exp in experiments:
            print(f"{exp['dataset_id']:<20} {exp['model_name']:<20} {exp['path']}")
        
        return True
    
    if args.generate_report:
        results_manager = get_results_manager()
        report = results_manager.create_summary_report(
            modality="tabular",
            output_file=args.generate_report
        )
        
        print(f"Generated report with {report['summary']['total_experiments']} experiments")
        print(f"Report saved to: {args.generate_report}")
        
        return True
    
    return False


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    from clam.utils import set_seed_with_args
    set_seed_with_args(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"llm_baseline_evaluation_{timestamp}.log"
    logger = setup_logging(log_file=os.path.join(args.output_dir, log_filename))
    logger.info(f"Arguments: {args}")
    
    # Handle special commands
    if handle_special_commands(args, logger):
        return
    
    # Initialize Weights & Biases with GPU monitoring if requested
    gpu_monitor = None
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("Weights & Biases requested but not installed. Run 'pip install wandb' to install.")
        else:
            if args.wandb_name is None:
                args.wandb_name = f"llm_baselines_{timestamp}"
            
            gpu_monitor = init_wandb_with_gpu_monitoring(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config=vars(args),
                output_dir=args.output_dir,
                enable_system_monitoring=True,
                gpu_log_interval=30.0,
                enable_detailed_gpu_logging=True
            )
            logger.info(f"Initialized Weights & Biases run with GPU monitoring: {args.wandb_name}")
    
    # Load datasets
    datasets = load_datasets(args)
    if not datasets:
        logger.error("No datasets loaded successfully. Exiting.")
        return
    
    # Parse models to evaluate (already a list from nargs="+")
    models_to_evaluate = args.models
    logger.info(f"Evaluating models: {models_to_evaluate}")
    
    # Evaluate each model on each dataset
    all_results = []
    
    for dataset in datasets:
        logger.info(f"\\n{'='*50}\\nEvaluating dataset: {dataset['name']}\\n{'='*50}")
        
        dataset_results = []
        
        # Simplified evaluation loop (using just one model as example)
        for model_name in models_to_evaluate[:1]:  # Just first model for demo
            logger.info(f"Evaluating {model_name} on {dataset['name']}")
            
            try:
                # Set timeout for each model evaluation
                timeout_seconds = args.timeout_minutes * 60
                
                with timeout_context(timeout_seconds):
                    # Simulate evaluation result (replace with actual evaluation)
                    result = {
                        'model_name': model_name,
                        'dataset_name': dataset['name'],
                        'dataset_id': dataset['id'],
                        'accuracy': 0.8500 + np.random.normal(0, 0.05),  # Simulated
                        'precision_macro': 0.8300 + np.random.normal(0, 0.03),
                        'recall_macro': 0.8400 + np.random.normal(0, 0.03),
                        'f1_macro': 0.8350 + np.random.normal(0, 0.03),
                        'completion_rate': 0.98,
                        'training_time': 45.2,
                        'prediction_time': 12.8,
                        'num_test_samples': 200,
                        'num_classes': len(dataset.get('class_names', [])),
                        'class_names': dataset.get('class_names', [])
                    }
                
            except TimeoutError:
                logger.warning(f"Evaluation of {model_name} on {dataset['name']} timed out after {args.timeout_minutes} minutes")
                result = {
                    'model_name': model_name,
                    'dataset_name': dataset['name'],
                    'dataset_id': dataset['id'],
                    'error': f'Timeout after {args.timeout_minutes} minutes',
                    'timeout': True
                }
            except Exception as e:
                logger.error(f"Error evaluating {model_name} on {dataset['name']}: {e}")
                result = {
                    'model_name': model_name,
                    'dataset_name': dataset['name'],
                    'dataset_id': dataset['id'],
                    'error': str(e)
                }
            
            dataset_results.append(result)
            all_results.append(result)
        
        # Save results for this dataset
        if args.use_unified_results:
            # Use new unified results manager
            logger.info("Saving results using unified results manager")
            save_results_unified_format(dataset_results, dataset, args, logger)
        else:
            # Use legacy results format
            logger.info("Saving results using legacy format")
            save_results(dataset_results, args.output_dir, dataset['name'])
    
    # Save aggregated results
    if not args.use_unified_results:
        # Only save aggregated results for legacy format
        # (unified format handles aggregation automatically)
        aggregated_file = os.path.join(args.output_dir, "aggregated_results.json")
        success = safe_json_dump(
            all_results, 
            aggregated_file, 
            logger=logger,
            minimal_fallback=True,
            indent=2
        )
        
        if success:
            print(f"Successfully saved aggregated results to {aggregated_file}")
        else:
            logger.error(f"Failed to save aggregated results to {aggregated_file}")
    
    # Print summary
    logger.info(f"\\n{'='*50}\\nEVALUATION SUMMARY\\n{'='*50}")
    
    for model_name in models_to_evaluate:
        model_results = [r for r in all_results if r.get('model_name') == model_name]
        if model_results:
            accuracies = [r['accuracy'] for r in model_results if 'accuracy' in r and r['accuracy'] is not None]
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                logger.info(f"{model_name}: Average accuracy = {avg_accuracy:.4f} ({len(accuracies)} datasets)")
    
    if args.use_unified_results:
        # Show information about unified results
        results_manager = get_results_manager()
        experiments = results_manager.list_experiments(modality="tabular")
        logger.info(f"\\nTotal tabular experiments in unified storage: {len(experiments)}")
        logger.info(f"Results base directory: {results_manager.get_results_base_dir()}")
    else:
        logger.info(f"\\nResults saved to: {args.output_dir}")
        logger.info(f"Aggregated results: {aggregated_file}")
    
    # Clean up GPU monitoring
    cleanup_gpu_monitoring(gpu_monitor)


if __name__ == "__main__":
    main()