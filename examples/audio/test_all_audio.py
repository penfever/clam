#!/usr/bin/env python3
"""
Comprehensive test script for audio classification datasets.

Runs few-shot audio classification on ESC-50, RAVDESS, and UrbanSound8K
using LLATA t-SNE with configurable audio embeddings (Whisper or CLAP).
Can test individual datasets or run comprehensive comparisons across multiple datasets.
"""

import argparse
import logging
import os
import sys
import time
import json
import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Import wandb conditionally
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from clam.utils.json_utils import convert_for_json_serialization
from clam.utils.platform_utils import log_platform_info
from clam.utils import (
    init_wandb_with_gpu_monitoring, 
    cleanup_gpu_monitoring,
    MetricsLogger
)

from examples.audio.clam_tsne_audio_baseline import ClamAudioTsneClassifier
from examples.audio.audio_datasets import ESC50Dataset, RAVDESSDataset, UrbanSound8KDataset
from examples.audio.audio_baselines import WhisperKNNClassifier, CLAPZeroShotClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_dataset(dataset_class, dataset_name, data_dir, args, use_wandb_logging=False):
    """Test a single audio dataset."""
    logger.info(f"\\n{'='*60}")
    logger.info(f"TESTING {dataset_name.upper()}")
    logger.info(f"{'='*60}")
    
    try:
        # Load dataset
        dataset = dataset_class(data_dir, download=not args.no_download)
        
        # Create few-shot splits
        splits = dataset.create_few_shot_split(
            k_shot=args.k_shot,
            val_size=0.2,
            test_size=0.3,
            random_state=42
        )
        
        train_paths, train_labels = splits['train']
        val_paths, val_labels = splits['val']
        test_paths, test_labels = splits['test']
        class_names = splits['class_names']
        
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"  Train: {len(train_paths)} samples ({args.k_shot} per class)")
        logger.info(f"  Val: {len(val_paths)} samples")
        logger.info(f"  Test: {len(test_paths)} samples")
        logger.info(f"  Classes: {len(class_names)} - {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
        
        # Use subset for quick testing
        if args.quick_test:
            logger.info("Running quick test with subset of data")
            test_paths = test_paths[:min(20, len(test_paths))]
            test_labels = test_labels[:min(20, len(test_labels))]
        
        # Configure audio duration based on dataset
        audio_duration = args.audio_duration
        if dataset_name.lower() == 'ravdess' and audio_duration is None:
            audio_duration = 3.0  # RAVDESS clips are ~3 seconds
        elif audio_duration is None:
            audio_duration = 5.0  # Default for ESC-50 and others
            
        results = {}
        
        # Test LLATA t-SNE
        if 'llata_tsne' in args.models:
            backend_name = "PCA" if args.use_pca_backend else "t-SNE"
            features = []
            if args.use_3d_tsne:
                features.append("3D")
            if args.use_knn_connections:
                features.append(f"KNN-{args.knn_k}")
            feature_str = f" ({', '.join(features)})" if features else ""
            
            logger.info(f"Testing LLATA {backend_name}{feature_str} ({args.embedding_model.upper()} → {backend_name} → VLM)...")
            
            try:
                classifier = ClamAudioTsneClassifier(
                    embedding_model=args.embedding_model,
                    whisper_model=args.whisper_model,
                    embedding_layer="encoder_last",
                    clap_version=args.clap_version,
                    tsne_perplexity=min(30.0, len(train_paths) / 4),
                    tsne_n_iter=1000,
                    vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                    use_3d_tsne=args.use_3d_tsne,
                    use_knn_connections=args.use_knn_connections,
                    knn_k=args.knn_k,
                    max_vlm_image_size=1024,
                    tsne_zoom_factor=args.tsne_zoom_factor,
                    use_pca_backend=args.use_pca_backend,
                    include_spectrogram=args.include_spectrogram,
                    audio_duration=audio_duration,
                    cache_dir=args.cache_dir,
                    use_semantic_names=args.use_semantic_names,
                    num_few_shot_examples=args.num_few_shot_examples,
                    balanced_few_shot=args.balanced_few_shot,
                    device='cpu' if sys.platform == "darwin" else None,
                    seed=42
                )
                
                # Pass save_every_n parameter
                classifier.save_every_n = args.save_every_n
                
                start_time = time.time()
                classifier.fit(train_paths, train_labels, test_paths[:5], class_names)
                training_time = time.time() - start_time
                
                eval_results = classifier.evaluate(
                    test_paths, test_labels, 
                    return_detailed=True,
                    save_outputs=args.save_outputs,
                    output_dir=os.path.join(args.output_dir, dataset_name.lower()) if args.save_outputs else None
                )
                eval_results['training_time'] = training_time
                eval_results['config'] = classifier.get_config()
                eval_results['dataset_info'] = {
                    'name': dataset_name,
                    'num_classes': len(class_names),
                    'class_names': class_names,
                    'train_samples': len(train_paths),
                    'test_samples': len(test_paths),
                    'k_shot': args.k_shot
                }
                
                results['llata_tsne'] = eval_results
                logger.info(f"{dataset_name} LLATA t-SNE completed: {eval_results['accuracy']:.4f} accuracy")
                
                # Log to wandb
                if use_wandb_logging:
                    log_results_to_wandb(f'{dataset_name.lower()}_llata_tsne', eval_results, args, class_names)
                
            except Exception as e:
                logger.error(f"{dataset_name} LLATA t-SNE failed: {e}")
                results['llata_tsne'] = {'error': str(e)}
        
        # Test Whisper KNN baseline
        if 'whisper_knn' in args.models:
            logger.info(f"Testing Whisper KNN baseline (Whisper → KNN)...")
            try:
                classifier = WhisperKNNClassifier(
                    whisper_model=args.whisper_model,
                    n_neighbors=5,
                    metric="cosine",
                    weights="distance",
                    standardize=True,
                    cache_dir=args.cache_dir,
                    device='cpu' if sys.platform == "darwin" else None,
                    seed=42
                )
                
                start_time = time.time()
                classifier.fit(train_paths, train_labels, class_names)
                training_time = time.time() - start_time
                
                eval_results = classifier.evaluate(
                    test_paths, test_labels,
                    return_detailed=True
                )
                eval_results['training_time'] = training_time
                eval_results['config'] = classifier.get_config()
                eval_results['dataset_info'] = {
                    'name': dataset_name,
                    'num_classes': len(class_names),
                    'class_names': class_names,
                    'train_samples': len(train_paths),
                    'test_samples': len(test_paths),
                    'k_shot': args.k_shot
                }
                
                results['whisper_knn'] = eval_results
                logger.info(f"{dataset_name} Whisper KNN completed: {eval_results['accuracy']:.4f} accuracy")
                
                # Log to wandb
                if use_wandb_logging:
                    log_results_to_wandb(f'{dataset_name.lower()}_whisper_knn', eval_results, args, class_names)
                
            except Exception as e:
                logger.error(f"{dataset_name} Whisper KNN failed: {e}")
                results['whisper_knn'] = {'error': str(e)}
        
        # Test CLAP zero-shot baseline
        if 'clap_zero_shot' in args.models:
            logger.info(f"Testing CLAP zero-shot baseline...")
            try:
                classifier = CLAPZeroShotClassifier(
                    model_name="microsoft/msclap",
                    device='cpu' if sys.platform == "darwin" else None,
                    cache_dir=args.cache_dir,
                    batch_size=4,  # Smaller batch for CPU
                    use_amp=False if sys.platform == "darwin" else True
                )
                
                start_time = time.time()
                classifier.fit(train_paths, train_labels, class_names)  # Only for class names
                training_time = time.time() - start_time
                
                eval_results = classifier.evaluate(
                    test_paths, test_labels,
                    return_detailed=True
                )
                eval_results['training_time'] = training_time
                eval_results['config'] = classifier.get_config()
                eval_results['dataset_info'] = {
                    'name': dataset_name,
                    'num_classes': len(class_names),
                    'class_names': class_names,
                    'train_samples': len(train_paths),
                    'test_samples': len(test_paths),
                    'k_shot': args.k_shot
                }
                
                results['clap_zero_shot'] = eval_results
                logger.info(f"{dataset_name} CLAP zero-shot completed: {eval_results['accuracy']:.4f} accuracy")
                
                # Log to wandb
                if use_wandb_logging:
                    log_results_to_wandb(f'{dataset_name.lower()}_clap_zero_shot', eval_results, args, class_names)
                
            except Exception as e:
                logger.error(f"{dataset_name} CLAP zero-shot failed: {e}")
                results['clap_zero_shot'] = {'error': str(e)}
        
        # Return first successful result for compatibility, or error if all failed
        if results:
            for model_name, result in results.items():
                if 'error' not in result:
                    return {
                        'status': 'success',
                        'results': result,
                        'all_results': results  # Include all results
                    }
            # All models failed
            return {
                'status': 'error',
                'error': f"All models failed",
                'all_results': results
            }
        else:
            return {
                'status': 'error',
                'error': f"No models specified for testing"
            }
        
    except Exception as e:
        logger.error(f"{dataset_name} test failed: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


def run_all_audio_tests(args):
    """Run tests on selected audio datasets."""
    all_results = {}
    
    # Store wandb availability for logging
    use_wandb_logging = args.use_wandb and WANDB_AVAILABLE
    
    # Define all available datasets
    available_datasets = {
        "esc50": (ESC50Dataset, "ESC-50", os.path.join(args.data_dir, "esc50")),
        "ravdess": (RAVDESSDataset, "RAVDESS", os.path.join(args.data_dir, "ravdess")),
        "urbansound8k": (UrbanSound8KDataset, "UrbanSound8K", os.path.join(args.data_dir, "urbansound8k"))
    }
    
    # Select datasets to test based on arguments
    datasets_to_test = []
    for dataset_key in args.datasets:
        if dataset_key in available_datasets:
            datasets_to_test.append(available_datasets[dataset_key])
        else:
            logger.warning(f"Unknown dataset: {dataset_key}. Available: {list(available_datasets.keys())}")
    
    if not datasets_to_test:
        logger.error("No valid datasets specified for testing")
        return {}
    
    # Test each selected dataset
    for dataset_class, dataset_name, data_dir in datasets_to_test:
        result = test_dataset(dataset_class, dataset_name, data_dir, args, use_wandb_logging)
        all_results[dataset_name.lower()] = result
    
    return all_results


def log_results_to_wandb(model_name: str, eval_results: dict, args, class_names: list):
    """Log evaluation results to Weights & Biases."""
    dataset_name = model_name.split('_')[0].upper()  # Extract dataset name
    
    if 'error' in eval_results:
        wandb.log({
            f"{model_name}/status": "failed",
            f"{model_name}/error": eval_results['error'],
            "model_name": model_name,
            "dataset": dataset_name,
            "k_shot": args.k_shot,
            "quick_test": args.quick_test
        })
        return
    
    # Create base metrics
    metrics = {
        f"{model_name}/accuracy": eval_results['accuracy'],
        f"{model_name}/training_time": eval_results.get('training_time', 0),
        f"{model_name}/prediction_time": eval_results.get('prediction_time', 0),
        f"{model_name}/num_test_samples": eval_results.get('num_test_samples', 0),
        "model_name": model_name,
        "dataset": dataset_name,
        "num_classes": len(class_names),
        "k_shot": args.k_shot,
        "quick_test": args.quick_test
    }
    
    # Add model-specific metrics
    config = eval_results.get('config', {})
    if 'llata_tsne' in model_name:
        # LLATA t-SNE specific metrics
        metrics.update({
            f"{model_name}/embedding_model": config.get('embedding_model', 'unknown'),
            f"{model_name}/whisper_model": config.get('whisper_model', 'unknown'),
            f"{model_name}/clap_version": config.get('clap_version', 'unknown'),
            f"{model_name}/use_3d_tsne": config.get('use_3d_tsne', False),
            f"{model_name}/use_knn_connections": config.get('use_knn_connections', False),
            f"{model_name}/knn_k": config.get('knn_k', 0),
            f"{model_name}/use_pca_backend": config.get('use_pca_backend', False),
            f"{model_name}/include_spectrogram": config.get('include_spectrogram', False),
            f"{model_name}/audio_duration": config.get('audio_duration', None),
            f"{model_name}/tsne_zoom_factor": config.get('tsne_zoom_factor', 1.0),
            f"{model_name}/vlm_model": config.get('vlm_model_id', 'unknown'),
        })
    elif 'whisper_knn' in model_name:
        # Whisper KNN specific metrics
        metrics.update({
            f"{model_name}/whisper_model": config.get('whisper_model', 'unknown'),
            f"{model_name}/n_neighbors": config.get('n_neighbors', 0),
            f"{model_name}/metric": config.get('metric', 'unknown'),
            f"{model_name}/weights": config.get('weights', 'unknown'),
            f"{model_name}/standardize": config.get('standardize', False),
        })
    elif 'clap_zero_shot' in model_name:
        # CLAP zero-shot specific metrics
        metrics.update({
            f"{model_name}/model_name": config.get('model_name', 'unknown'),
            f"{model_name}/batch_size": config.get('batch_size', 0),
            f"{model_name}/use_amp": config.get('use_amp', False),
        })
    
    # Add dataset info
    dataset_info = eval_results.get('dataset_info', {})
    if dataset_info:
        metrics.update({
            f"{model_name}/train_samples": dataset_info.get('train_samples', 0),
            f"{model_name}/test_samples": dataset_info.get('test_samples', 0),
        })
    
    # Add visualization info if available
    if eval_results.get('visualizations_saved', False):
        metrics[f"{model_name}/visualizations_saved"] = True
        metrics[f"{model_name}/output_directory"] = eval_results.get('output_directory', 'unknown')
    
    # Add classification report metrics if available
    if 'classification_report' in eval_results:
        class_report = eval_results['classification_report']
        if isinstance(class_report, dict):
            # Log macro/weighted averages
            for avg_type in ['macro avg', 'weighted avg']:
                if avg_type in class_report:
                    avg_metrics = class_report[avg_type]
                    avg_prefix = avg_type.replace(' ', '_')
                    metrics.update({
                        f"{model_name}/precision_{avg_prefix}": avg_metrics.get('precision', 0),
                        f"{model_name}/recall_{avg_prefix}": avg_metrics.get('recall', 0),
                        f"{model_name}/f1_{avg_prefix}": avg_metrics.get('f1-score', 0),
                    })
    
    # Log all metrics to wandb
    wandb.log(metrics)


def save_results(results: dict, output_dir: str, k_shot: int):
    """Save comprehensive test results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"all_audio_k{k_shot}_test_results.json")
    with open(results_file, 'w') as f:
        json_results = convert_for_json_serialization(results)
        json.dump(json_results, f, indent=2)
    
    # Create summary table
    summary_data = []
    for dataset_name, result in results.items():
        if result['status'] == 'success':
            eval_results = result['results']
            dataset_info = eval_results.get('dataset_info', {})
            summary_data.append({
                'dataset': dataset_name.upper(),
                'status': 'SUCCESS',
                'accuracy': eval_results['accuracy'],
                'num_classes': dataset_info.get('num_classes', 'N/A'),
                'train_samples': dataset_info.get('train_samples', 'N/A'),
                'test_samples': dataset_info.get('test_samples', 'N/A'),
                'training_time': eval_results['training_time'],
                'prediction_time': eval_results['prediction_time'],
                'error': None
            })
        else:
            summary_data.append({
                'dataset': dataset_name.upper(),
                'model': 'ALL',
                'status': 'ERROR',
                'accuracy': None,
                'num_classes': None,
                'train_samples': None,
                'test_samples': None,
                'training_time': None,
                'prediction_time': None,
                'error': result['error']
            })
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f"all_audio_k{k_shot}_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Print comprehensive summary
    logger.info("\\n" + "="*80)
    logger.info(f"COMPREHENSIVE AUDIO CLASSIFICATION RESULTS (k={k_shot})")
    logger.info("="*80)
    
    success_count = 0
    total_datasets = len(results)
    
    for _, row in summary_df.iterrows():
        if row['status'] == 'SUCCESS':
            success_count += 1
            logger.info(f"{row['dataset']:12s} {row['model']:15s}: ✓ {row['accuracy']:.4f} accuracy "
                       f"({row['num_classes']} classes, {row['train_samples']} train, {row['test_samples']} test) "
                       f"[Train: {row['training_time']:.1f}s, Test: {row['prediction_time']:.1f}s]")
        else:
            logger.info(f"{row['dataset']:12s} {row['model']:15s}: ✗ ERROR - {row['error']}")
    
    logger.info("\\n" + "-"*80)
    logger.info(f"SUMMARY: {success_count}/{len(summary_data)} experiments successful across {total_datasets} datasets")
    
    if success_count > 0:
        successful_results = summary_df[summary_df['status'] == 'SUCCESS']
        mean_accuracy = successful_results['accuracy'].mean()
        logger.info(f"Mean accuracy across successful experiments: {mean_accuracy:.4f}")
        
        # Show per-model averages if multiple models tested
        if 'model' in summary_df.columns and len(summary_df['model'].unique()) > 1:
            logger.info("\\nPer-model averages:")
            for model in summary_df['model'].unique():
                model_results = successful_results[successful_results['model'] == model]
                if len(model_results) > 0:
                    model_mean = model_results['accuracy'].mean()
                    logger.info(f"  {model:15s}: {model_mean:.4f} (across {len(model_results)} datasets)")
    
    logger.info(f"\\nDetailed results saved to: {output_dir}")
    logger.info("="*80)


def parse_args():
    parser = argparse.ArgumentParser(description="Test audio datasets with LLATA t-SNE and baselines")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./audio_data",
        help="Base directory for all audio datasets"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory for caching embeddings"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./all_audio_test_results",
        help="Directory for test results"
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=5,
        help="Number of training examples per class for few-shot learning (e.g., k_shot=5 means 5 samples per class for training)"
    )
    parser.add_argument(
        "--num_few_shot_examples",
        type=int,
        default=32,
        help="Number of examples to use for in-context learning in LLM prompts (for LLATA baseline)"
    )
    parser.add_argument(
        "--balanced_few_shot",
        action="store_true",
        help="Use class-balanced few-shot examples in LLM prompts instead of random selection"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="whisper",
        choices=["whisper", "clap"],
        help="Audio embedding model to use (whisper or clap)"
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="large-v2",
        choices=["tiny", "base", "small", "medium", "large", "large-v2"],
        help="Whisper model size (used if embedding_model is whisper)"
    )
    parser.add_argument(
        "--clap_version",
        type=str,
        default="2023",
        choices=["2022", "2023", "clapcap"],
        help="CLAP model version (used if embedding_model is clap)"
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run quick test with subset of data"
    )
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="Skip automatic dataset downloads"
    )
    parser.add_argument(
        "--tsne_zoom_factor",
        type=float,
        default=4.0,
        help="Zoom factor for t-SNE visualizations"
    )
    parser.add_argument(
        "--use_pca_backend",
        action="store_true",
        help="Use PCA instead of t-SNE for dimensionality reduction"
    )
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        default=True,
        help="Save visualizations and VLM responses (default: True)"
    )
    parser.add_argument(
        "--no_save_outputs",
        dest="save_outputs",
        action="store_false",
        help="Disable saving visualizations and VLM responses"
    )
    parser.add_argument(
        "--use_knn_connections",
        action="store_true",
        help="Show KNN connections from query point to nearest neighbors"
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=5,
        help="Number of nearest neighbors to show when using KNN connections"
    )
    parser.add_argument(
        "--use_3d_tsne",
        action="store_true",
        help="Use 3D t-SNE with multiple viewing angles instead of 2D"
    )
    parser.add_argument(
        "--include_spectrogram",
        action="store_true",
        default=True,
        help="Include spectrogram in visualization"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["clam_tsne"],
        choices=["clam_tsne", "whisper_knn", "clap_zero_shot"],
        help="Models to test (default: llata_tsne)"
    )
    parser.add_argument(
        "--save_every_n",
        type=int,
        default=10,
        help="Save visualizations every N predictions"
    )
    parser.add_argument(
        "--audio_duration",
        type=float,
        default=None,
        help="Maximum audio duration to process (seconds, auto-detected per dataset if None)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["esc50", "ravdess"],
        choices=["esc50", "ravdess", "urbansound8k"],
        help="Datasets to test (default: esc50, ravdess). Use 'urbansound8k' to include UrbanSound8K"
    )
    parser.add_argument(
        "--use_semantic_names",
        action="store_true",
        help="Use semantic class names in prompts instead of 'Class X' format"
    )
    
    # Weights & Biases logging
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="audio-llata-all",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity name"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("Starting audio classification test...")
    logger.info(f"Configuration:")
    logger.info(f"  Datasets: {', '.join(args.datasets)}")
    logger.info(f"  Models: {', '.join(args.models)}")
    logger.info(f"  k-shot: {args.k_shot}")
    logger.info(f"  Embedding model: {args.embedding_model}")
    if args.embedding_model == "whisper":
        logger.info(f"  Whisper model: {args.whisper_model}")
    elif args.embedding_model == "clap":
        logger.info(f"  CLAP version: {args.clap_version}")
    logger.info(f"  Quick test: {args.quick_test}")
    logger.info(f"  Use PCA: {args.use_pca_backend}")
    logger.info(f"  3D t-SNE: {args.use_3d_tsne}")
    logger.info(f"  Include spectrogram: {args.include_spectrogram}")
    
    # Initialize Weights & Biases with GPU monitoring if requested
    gpu_monitor = None
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("Weights & Biases requested but not installed. Run 'pip install wandb' to install.")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.wandb_name is None:
                datasets_str = "_".join(args.datasets)
                feature_suffix = f"_k{args.k_shot}_{datasets_str}"
                if args.use_3d_tsne:
                    feature_suffix += "_3d"
                if args.use_knn_connections:
                    feature_suffix += f"_knn{args.knn_k}"
                if args.use_pca_backend:
                    feature_suffix += "_pca"
                args.wandb_name = f"audio_llata_{timestamp}{feature_suffix}"
            
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
            logger.info(f"Initialized Weights & Biases run: {args.wandb_name}")
    
    # Log platform information
    platform_info = log_platform_info(logger)
    
    # Run tests
    start_time = time.time()
    results = run_all_audio_tests(args)
    total_time = time.time() - start_time
    
    # Save results
    save_results(results, args.output_dir, args.k_shot)
    
    # Log summary to wandb
    if args.use_wandb and WANDB_AVAILABLE:
        # Count successful experiments
        success_count = 0
        for r in results.values():
            if r.get('status') == 'success':
                if 'all_results' in r:
                    success_count += sum(1 for model_result in r['all_results'].values() if 'error' not in model_result)
                else:
                    success_count += 1
        
        total_datasets = len(results)
        total_experiments = sum(len(r.get('all_results', ['default'])) for r in results.values())
        
        wandb.log({
            "summary/datasets_tested": total_datasets,
            "summary/experiments_successful": success_count,
            "summary/total_experiments": total_experiments,
            "summary/total_test_time": total_time,
            "summary/k_shot": args.k_shot,
        })
        
        # Log mean accuracy across successful experiments
        accuracies = []
        for r in results.values():
            if r.get('status') == 'success':
                if 'results' in r:
                    accuracies.append(r['results']['accuracy'])
                if 'all_results' in r:
                    for model_result in r['all_results'].values():
                        if 'error' not in model_result:
                            accuracies.append(model_result['accuracy'])
        if accuracies:
            wandb.log({"summary/mean_accuracy": np.mean(accuracies)})
    
    # Clean up wandb
    if gpu_monitor is not None:
        cleanup_gpu_monitoring(gpu_monitor)
    
    logger.info(f"\\nTotal test time: {total_time:.1f} seconds")
    logger.info(f"Audio tests completed for datasets: {', '.join(args.datasets)}")


if __name__ == "__main__":
    main()