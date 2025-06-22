#!/usr/bin/env python3
"""
Script to parse OpenML CC18 results from tar archives and generate summary spreadsheets.

This script:
1. Extracts and parses results from multiple tar archives containing OpenML CC18 runs
2. Generates two summary spreadsheets:
   - Aggregated performance across all datasets with 95% confidence intervals
   - Per-dataset performance with confidence intervals
3. Handles missing results by imputing chance-level performance
4. Rounds all values to 3 significant digits

Usage:
    python parse_openml_cc18_results.py --results_dir /path/to/results --output_dir /path/to/output
"""

import argparse
import json
import logging
import os
import tarfile
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def round_to_n_significant_digits(x: float, n: int = 3) -> float:
    """Round a number to n significant digits."""
    if x == 0:
        return 0.0
    return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for a list of values using normal approximation."""
    if not values or len(values) < 2:
        return np.nan, np.nan
    
    mean = np.mean(values)
    std_err = stats.sem(values)  # Standard error of the mean
    
    # Use t-distribution for small samples
    if len(values) < 30:
        dof = len(values) - 1
        t_critical = stats.t.ppf((1 + confidence) / 2, dof)
        margin_error = t_critical * std_err
    else:
        # Use normal distribution for large samples
        z_critical = stats.norm.ppf((1 + confidence) / 2)
        margin_error = z_critical * std_err
    
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return ci_lower, ci_upper


def extract_results_from_tar(tar_path: str, temp_dir: str) -> List[Dict[str, Any]]:
    """Extract and parse results from a tar archive."""
    logger = logging.getLogger(__name__)
    results = []
    archive_name = os.path.basename(tar_path).replace('.tar', '')
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # Look for all_results_*.json files first (consolidated format)
            all_results_files = [name for name in tar.getnames() 
                               if name.endswith('.json') and 'all_results_' in name]
            
            # If no consolidated files, look for individual aggregated_results.json files
            if not all_results_files:
                aggregated_files = [name for name in tar.getnames() 
                                  if name.endswith('aggregated_results.json')]
                if aggregated_files:
                    logger.info(f"Found {len(aggregated_files)} individual aggregated_results.json files in {archive_name}")
                    all_results_files = aggregated_files
                else:
                    logger.warning(f"No all_results_*.json or aggregated_results.json files found in {tar_path}")
                    return results
            
            for file_name in all_results_files:
                try:
                    # Extract the file to temp directory
                    tar.extract(file_name, temp_dir)
                    extracted_path = os.path.join(temp_dir, file_name)
                    
                    # Load and parse JSON
                    with open(extracted_path, 'r') as f:
                        data = json.load(f)
                    
                    # Add archive source information to each result
                    if isinstance(data, list):
                        for result in data:
                            if isinstance(result, dict):
                                result['_archive_source'] = archive_name
                        results.extend(data)
                    elif isinstance(data, dict):
                        data['_archive_source'] = archive_name
                        results.append(data)
                    
                    logger.info(f"Loaded {len(data) if isinstance(data, list) else 1} results from {file_name}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error extracting {tar_path}: {e}")
    
    return results


def get_chance_level_performance(n_classes: int) -> Dict[str, float]:
    """Calculate chance-level performance metrics for a given number of classes."""
    chance_accuracy = 1.0 / n_classes
    chance_balanced_accuracy = 1.0 / n_classes
    chance_f1_macro = 1.0 / n_classes  # Simplified approximation
    # ROC AUC is not imputed as requested
    
    return {
        'accuracy': chance_accuracy,
        'balanced_accuracy': chance_balanced_accuracy,
        'f1_macro': chance_f1_macro,
        'f1_micro': chance_accuracy,  # Same as accuracy for balanced case
        'f1_weighted': chance_f1_macro,
        'precision_macro': chance_accuracy,
        'recall_macro': chance_accuracy,
        'roc_auc': None  # Do not impute ROC AUC as requested
    }


def normalize_model_name(model_name: str) -> str:
    """Normalize model names for consistency."""
    model_name = model_name.lower().strip()
    
    # Map variations to standard names
    name_mapping = {
        'clam-t-sne-tabular': 'clam_tsne',
        'clam_t_sne_tabular': 'clam_tsne',
        'clam-tsne': 'clam_tsne',
        'clam-t-sne': 'clam_tsne',  # Fix missing mapping
        'clam_tsne': 'clam_tsne',   # Add direct mapping for consistency
        'jolt': 'jolt',
        'tabllm': 'tabllm',
        'tabula-8b': 'tabula_8b',
        'tabula_8b': 'tabula_8b'
    }
    
    return name_mapping.get(model_name, model_name)


def create_unique_model_identifier(model_name: str, archive_source: str, model_used: str = None) -> str:
    """Create a unique model identifier based on model name, archive source, and backend."""
    normalized_name = normalize_model_name(model_name)
    
    # For tabllm, try to extract backend information from model_used field
    if normalized_name == 'tabllm' and model_used:
        if 'qwen' in model_used.lower():
            return 'tabllm_qwen'
        elif 'llama' in model_used.lower():
            return 'tabllm_llama'
        elif 'mistral' in model_used.lower():
            return 'tabllm_mistral'
        elif 'gemma' in model_used.lower():
            return 'tabllm_gemma'
        elif 'gpt' in model_used.lower():
            return 'tabllm_gpt'
        elif 'gemini' in model_used.lower():
            return 'tabllm_gemini'
    
    # If we can't distinguish by model_used, use archive source as disambiguator
    # Extract meaningful parts from archive name
    archive_lower = archive_source.lower()
    
    # Look for LLM indicators in archive name
    if normalized_name == 'tabllm':
        if 'qwen' in archive_lower:
            return 'tabllm_qwen'
        elif 'llama' in archive_lower:
            return 'tabllm_llama' 
        elif 'mistral' in archive_lower:
            return 'tabllm_mistral'
        elif 'gemma' in archive_lower:
            return 'tabllm_gemma'
        elif 'gpt' in archive_lower or 'openai' in archive_lower:
            return 'tabllm_gpt'
        elif 'gemini' in archive_lower:
            return 'tabllm_gemini'
        elif '32b' in archive_lower:
            return 'tabllm_32b'
        elif '8b' in archive_lower:
            return 'tabllm_8b'
        elif '3b' in archive_lower:
            return 'tabllm_3b'
        else:
            # Fall back to archive-based suffix
            return f'tabllm_{archive_lower.replace("_", "").replace("-", "")}'
    
    return normalized_name


def detect_model_conflicts(all_results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Detect potential model name conflicts across archives."""
    model_archive_mapping = defaultdict(set)
    
    for result in all_results:
        model_name = normalize_model_name(result.get('model_name', 'unknown'))
        archive_source = result.get('_archive_source', 'unknown')
        model_archive_mapping[model_name].add(archive_source)
    
    # Find models that appear in multiple archives
    conflicts = {model: list(archives) for model, archives in model_archive_mapping.items() 
                if len(archives) > 1}
    
    return conflicts


def process_results(all_results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process results and generate summary dataframes."""
    logger = logging.getLogger(__name__)
    
    # Detect and report model conflicts
    conflicts = detect_model_conflicts(all_results)
    if conflicts:
        logger.info("Detected model name conflicts across archives:")
        for model, archives in conflicts.items():
            logger.info(f"  {model}: found in archives {', '.join(archives)}")
        logger.info("Using unique identifiers to distinguish models...")
    
    # Organize results by model, dataset, and task
    organized_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    dataset_info = {}  # Store dataset metadata
    
    for result in all_results:
        original_model_name = result.get('model_name', 'unknown')
        archive_source = result.get('_archive_source', 'unknown')
        model_used = result.get('model_used', None)
        
        # Create unique model identifier
        unique_model_name = create_unique_model_identifier(
            original_model_name, archive_source, model_used
        )
        
        # Log unique identifier creation for first occurrence
        if unique_model_name != normalize_model_name(original_model_name):
            logger.debug(f"Created unique identifier: {original_model_name} -> {unique_model_name} (archive: {archive_source})")
        
        dataset_name = result.get('dataset_name', 'unknown')
        task_id = str(result.get('task_id', result.get('dataset_id', 'unknown')))
        
        # Store dataset info
        if task_id not in dataset_info:
            dataset_info[task_id] = {
                'dataset_name': dataset_name,
                'n_classes': result.get('num_classes', 2),
                'task_type': result.get('task_type', 'classification')
            }
        
        # Extract key metrics
        metrics = {
            'accuracy': result.get('accuracy'),
            'balanced_accuracy': result.get('balanced_accuracy'),
            'f1_macro': result.get('f1_macro'),
            'f1_micro': result.get('f1_micro'),
            'f1_weighted': result.get('f1_weighted'),
            'precision_macro': result.get('precision_macro'),
            'recall_macro': result.get('recall_macro'),
            'roc_auc': result.get('roc_auc'),
            'completion_rate': result.get('completion_rate', 1.0)
        }
        
        organized_results[unique_model_name][task_id]['metrics'].append(metrics)
        
        # Store result details for per-dataset analysis (include model metadata)
        result_with_metadata = result.copy()
        result_with_metadata['_unique_model_name'] = unique_model_name
        result_with_metadata['_original_model_name'] = original_model_name
        organized_results[unique_model_name][task_id]['raw_results'].append(result_with_metadata)
    
    logger.info(f"Processed {len(all_results)} results across {len(organized_results)} models and {len(dataset_info)} datasets")
    
    # Generate aggregated summary
    aggregated_data = []
    per_dataset_data = []
    
    for model_name, model_results in organized_results.items():
        # Collect all metrics across datasets for this model
        all_model_metrics = defaultdict(list)
        
        # Extract metadata from first result to get original model name and archive source
        first_result = None
        for task_results in model_results.values():
            if task_results['raw_results']:
                first_result = task_results['raw_results'][0]
                break
        
        original_model_name = first_result.get('_original_model_name', model_name) if first_result else model_name
        archive_source = first_result.get('_archive_source', 'unknown') if first_result else 'unknown'
        model_used = first_result.get('model_used', None) if first_result else None
        
        for task_id, task_results in model_results.items():
            dataset_name = dataset_info[task_id]['dataset_name']
            n_classes = dataset_info[task_id]['n_classes']
            
            # Get metrics for this dataset
            metrics_list = task_results['metrics']
            
            if not metrics_list:
                # No results for this dataset - use chance level
                chance_metrics = get_chance_level_performance(n_classes)
                metrics_list = [chance_metrics]
                logger.warning(f"No results for {model_name} on {dataset_name} (task {task_id}), using chance level")
            
            # Calculate per-dataset statistics
            dataset_stats = {}
            for metric_name in ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_micro', 
                              'f1_weighted', 'precision_macro', 'recall_macro', 'roc_auc']:
                values = [m.get(metric_name) for m in metrics_list if m.get(metric_name) is not None]
                
                if values:
                    mean_val = np.mean(values)
                    ci_lower, ci_upper = calculate_confidence_interval(values)
                    
                    dataset_stats[f'{metric_name}_mean'] = round_to_n_significant_digits(mean_val)
                    dataset_stats[f'{metric_name}_ci_lower'] = round_to_n_significant_digits(ci_lower) if not np.isnan(ci_lower) else np.nan
                    dataset_stats[f'{metric_name}_ci_upper'] = round_to_n_significant_digits(ci_upper) if not np.isnan(ci_upper) else np.nan
                    dataset_stats[f'{metric_name}_std'] = round_to_n_significant_digits(np.std(values))
                    dataset_stats[f'{metric_name}_n_runs'] = len(values)
                    
                    # Add to model aggregation
                    all_model_metrics[metric_name].extend(values)
                else:
                    # No valid values for this metric
                    dataset_stats[f'{metric_name}_mean'] = np.nan
                    dataset_stats[f'{metric_name}_ci_lower'] = np.nan
                    dataset_stats[f'{metric_name}_ci_upper'] = np.nan
                    dataset_stats[f'{metric_name}_std'] = np.nan
                    dataset_stats[f'{metric_name}_n_runs'] = 0
            
            # Add to per-dataset results with metadata
            per_dataset_data.append({
                'model': model_name,
                'original_model_name': original_model_name,
                'archive_source': archive_source,
                'model_used': model_used,
                'task_id': task_id,
                'dataset_name': dataset_name,
                'n_classes': n_classes,
                **dataset_stats
            })
        
        # Calculate aggregated statistics across all datasets for this model
        agg_stats = {
            'model': model_name,
            'original_model_name': original_model_name,
            'archive_source': archive_source,
            'model_used': model_used
        }
        
        for metric_name in ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_micro', 
                          'f1_weighted', 'precision_macro', 'recall_macro', 'roc_auc']:
            values = all_model_metrics[metric_name]
            
            if values:
                mean_val = np.mean(values)
                median_val = np.median(values)
                ci_lower, ci_upper = calculate_confidence_interval(values)
                
                agg_stats[f'{metric_name}_mean'] = round_to_n_significant_digits(mean_val)
                agg_stats[f'{metric_name}_median'] = round_to_n_significant_digits(median_val)
                agg_stats[f'{metric_name}_ci_lower'] = round_to_n_significant_digits(ci_lower) if not np.isnan(ci_lower) else np.nan
                agg_stats[f'{metric_name}_ci_upper'] = round_to_n_significant_digits(ci_upper) if not np.isnan(ci_upper) else np.nan
                agg_stats[f'{metric_name}_std'] = round_to_n_significant_digits(np.std(values))
                agg_stats[f'{metric_name}_n_datasets'] = len([task_id for task_id, task_results in model_results.items() 
                                                              if task_results['metrics']])
                agg_stats[f'{metric_name}_n_runs'] = len(values)
            else:
                agg_stats[f'{metric_name}_mean'] = np.nan
                agg_stats[f'{metric_name}_median'] = np.nan
                agg_stats[f'{metric_name}_ci_lower'] = np.nan
                agg_stats[f'{metric_name}_ci_upper'] = np.nan
                agg_stats[f'{metric_name}_std'] = np.nan
                agg_stats[f'{metric_name}_n_datasets'] = 0
                agg_stats[f'{metric_name}_n_runs'] = 0
        
        aggregated_data.append(agg_stats)
    
    # Convert to dataframes
    aggregated_df = pd.DataFrame(aggregated_data)
    per_dataset_df = pd.DataFrame(per_dataset_data)
    
    return aggregated_df, per_dataset_df


def save_results(aggregated_df: pd.DataFrame, per_dataset_df: pd.DataFrame, output_dir: str):
    """Save results to Excel files with proper formatting."""
    logger = logging.getLogger(__name__)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save aggregated results
    agg_path = os.path.join(output_dir, 'openml_cc18_aggregated_results.xlsx')
    with pd.ExcelWriter(agg_path, engine='openpyxl') as writer:
        aggregated_df.to_excel(writer, sheet_name='Aggregated_Results', index=False)
    
    logger.info(f"Saved aggregated results to {agg_path}")
    
    # Save per-dataset results
    dataset_path = os.path.join(output_dir, 'openml_cc18_per_dataset_results.xlsx')
    with pd.ExcelWriter(dataset_path, engine='openpyxl') as writer:
        per_dataset_df.to_excel(writer, sheet_name='Per_Dataset_Results', index=False)
    
    logger.info(f"Saved per-dataset results to {dataset_path}")
    
    # Also save as CSV for easier analysis
    aggregated_df.to_csv(os.path.join(output_dir, 'openml_cc18_aggregated_results.csv'), index=False)
    per_dataset_df.to_csv(os.path.join(output_dir, 'openml_cc18_per_dataset_results.csv'), index=False)
    
    logger.info("Also saved CSV versions of both files")


def main():
    parser = argparse.ArgumentParser(description="Parse OpenML CC18 results from tar archives")
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="../results",
        help="Directory containing tar archives with results"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../results/analysis",
        help="Directory to save summary spreadsheets"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    logger = setup_logging(args.log_level)
    
    # Convert paths to absolute
    results_dir = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    logger.info(f"Processing results from: {results_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Find tar archives
    tar_files = []
    for file_name in os.listdir(results_dir):
        if file_name.endswith('.tar'):
            tar_path = os.path.join(results_dir, file_name)
            tar_files.append(tar_path)
            logger.info(f"Found tar archive: {file_name}")
    
    if not tar_files:
        logger.error(f"No tar files found in {results_dir}")
        return
    
    # Extract and process results
    all_results = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for tar_path in tar_files:
            logger.info(f"Processing {os.path.basename(tar_path)}...")
            try:
                results = extract_results_from_tar(tar_path, temp_dir)
                all_results.extend(results)
                logger.info(f"Extracted {len(results)} results from {os.path.basename(tar_path)}")
            except Exception as e:
                logger.error(f"Error processing {tar_path}: {e}")
                logger.debug(traceback.format_exc())
    
    if not all_results:
        logger.error("No results found in any tar archives")
        return
    
    logger.info(f"Total results collected: {len(all_results)}")
    
    # Process results and generate summaries
    try:
        aggregated_df, per_dataset_df = process_results(all_results)
        
        logger.info(f"Generated aggregated summary with {len(aggregated_df)} model entries")
        logger.info(f"Generated per-dataset summary with {len(per_dataset_df)} entries")
        
        # Display summary
        print("\\n" + "="*80)
        print("AGGREGATED RESULTS SUMMARY")
        print("="*80)
        print(aggregated_df[['model', 'accuracy_mean', 'accuracy_ci_lower', 'accuracy_ci_upper', 
                           'f1_macro_mean', 'f1_macro_ci_lower', 'f1_macro_ci_upper']].to_string(index=False))
        
        # Save results
        save_results(aggregated_df, per_dataset_df, output_dir)
        
        logger.info("Analysis complete!")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    main()