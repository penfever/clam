#!/usr/bin/env python3
"""
Analyze correlation between MARVIS and TabPFN v2 performance on regression and classification datasets.

This script creates correlation plots for both regression and classification tasks,
showing how MARVIS and TabPFN v2 performance relate across different datasets.

Usage:
    python scripts/analyze_clam_tabpfn_correlation.py --input_dir /path/to/results --output_dir /path/to/output
"""

import argparse
import json
import tarfile
import tempfile
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple, Optional
import statistics

def load_tabpfnv2_supplemental_results(supplemental_path: str) -> Dict[str, Dict]:
    """Load TabPFNv2 supplemental results from separate JSON file for regression."""
    supplemental_results = {}
    
    if not Path(supplemental_path).exists():
        print(f"⚠️  TabPFNv2 supplemental results not found: {supplemental_path}")
        return supplemental_results
    
    # Mapping from friendly names to task IDs (from regression analysis script)
    name_to_task_mapping = {
        'Crime_Data_from_2010': 'task_363432',
        'SGEMM_GPU_kernel_performance': 'task_361104',
        'convai2_inferred': 'task_363387',
        'jigsaw-unintended-bias-in-toxicity': 'task_363391',
        'light_inferred': 'task_363388',
        'medical_charges': 'task_361086',
        'mercari_price_suggestion100K': 'task_363377',
        'methane_emissions_rice_crop': 'task_363444',
        'opensubtitles_inferred': 'task_363389',
        'particulate-matter-ukair-2017': 'task_361103',
        'world_food_wealth_bank': 'task_363442'
    }
    
    try:
        with open(supplemental_path, 'r') as f:
            data = json.load(f)
        
        print(f"📦 Processing TabPFNv2 supplemental results from {supplemental_path}")
        print(f"  Found {len(data)} supplemental TabPFNv2 results")
        
        for entry in data:
            if entry.get('model_name') == 'tabpfn_v2' and entry.get('task_type') == 'regression':
                dataset_name = entry.get('dataset_name', 'unknown')
                r2_score = entry.get('r2', None)
                
                # Map to task_id if available
                task_id = name_to_task_mapping.get(dataset_name)
                if task_id and r2_score is not None:
                    supplemental_results[task_id] = {
                        'algorithm': 'tabpfn_v2',
                        'dataset_name': task_id,
                        'task_id': task_id,
                        'r2_score': r2_score,
                        'mae': entry.get('mae', None),
                        'rmse': entry.get('rmse', None)
                    }
                    print(f"    Mapped {dataset_name} -> {task_id}")
        
        print(f"    Loaded {len(supplemental_results)} TabPFNv2 replacement results")
        
    except Exception as e:
        print(f"⚠️  Error loading TabPFNv2 supplemental results: {e}")
    
    return supplemental_results

def extract_regression_results_from_tar(tar_path: str, temp_dir: str) -> List[Dict]:
    """Extract regression results from tar archives."""
    results = []
    
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(temp_dir)
    
    temp_path = Path(temp_dir)
    
    # Pattern 1: MARVIS results - look for actual metrics files
    for marvis_file in temp_path.glob('**/task_*/split_*/llm_baselines/dataset_*/clam_t_sne_tabular_results.json'):
        parts = marvis_file.parts
        task_id = None
        split_id = None
        
        for part in parts:
            if part.startswith('task_'):
                task_id = part
            elif part.startswith('split_'):
                split_id = part
        
        if task_id and split_id:
            try:
                with open(marvis_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    r2_score = data.get('r2_score', data.get('r2', None))
                    if r2_score is not None:
                        results.append({
                            'algorithm': 'MARVIS',
                            'dataset_name': task_id,
                            'task_id': task_id,
                            'split_id': split_id,
                            'r2_score': r2_score,
                            'mae': data.get('mae', None),
                            'rmse': data.get('rmse', None)
                        })
            except Exception as e:
                print(f"⚠️  Error processing {marvis_file}: {e}")
    
    # Pattern 2: Tabular baselines - task_*/split_*/baselines/all_evaluation_results_*.json
    for eval_file in temp_path.glob('**/task_*/split_*/baselines/all_evaluation_results_*.json'):
        parts = eval_file.parts
        task_id = None
        split_id = None
        
        for part in parts:
            if part.startswith('task_'):
                task_id = part
            elif part.startswith('split_'):
                split_id = part
        
        if task_id and split_id:
            try:
                with open(eval_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for entry in data:
                        model_name = entry.get('model_name', entry.get('algorithm', entry.get('model', 'baseline')))
                        r2_score = entry.get('r2_score', entry.get('r2', entry.get('test_r2', None)))
                        
                        if r2_score is not None:
                            results.append({
                                'algorithm': model_name,
                                'dataset_name': task_id,
                                'task_id': task_id,
                                'split_id': split_id,
                                'r2_score': r2_score,
                                'mae': entry.get('mae', entry.get('test_mae', None)),
                                'rmse': entry.get('rmse', entry.get('test_rmse', None))
                            })
            except Exception as e:
                print(f"⚠️  Error processing {eval_file}: {e}")
    
    return results

def extract_classification_results_from_tar(tar_path: str, temp_dir: str) -> List[Dict]:
    """Extract classification results from tar archives using patterns from parse_openml_cc18_results.py."""
    results = []
    archive_name = Path(tar_path).stem
    
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
                    all_results_files = aggregated_files
                else:
                    # Look for tabular baseline files (all_evaluation_results_*.json)
                    eval_results_files = [name for name in tar.getnames() 
                                        if name.endswith('.json') and 'all_evaluation_results_' in name]
                    if eval_results_files:
                        all_results_files = eval_results_files
            
            for file_name in all_results_files:
                try:
                    # Extract the file to temp directory
                    tar.extract(file_name, temp_dir)
                    extracted_path = Path(temp_dir) / file_name
                    
                    # Load and parse JSON
                    with open(extracted_path, 'r') as f:
                        data = json.load(f)
                    
                    # Handle tabular baseline format
                    if 'all_evaluation_results_' in file_name:
                        # Process tabular baseline results
                        processed_results = process_tabular_baseline_results(data, file_name, archive_name)
                        results.extend(processed_results)
                    else:
                        # Handle existing formats
                        if isinstance(data, list):
                            for result in data:
                                if isinstance(result, dict):
                                    result['_archive_source'] = archive_name
                            results.extend(data)
                        elif isinstance(data, dict):
                            data['_archive_source'] = archive_name
                            results.append(data)
                    
                except Exception as e:
                    print(f"⚠️  Error processing {file_name}: {e}")
                    continue
    
    except Exception as e:
        print(f"⚠️  Error extracting {tar_path}: {e}")
    
    return results

def process_tabular_baseline_results(data: List[Dict], file_name: str, archive_name: str) -> List[Dict]:
    """Process tabular baseline results into standard format."""
    processed_results = []
    
    # Extract task information from file path
    path_parts = file_name.split('/')
    task_id = None
    split_id = None
    
    for part in path_parts:
        if part.startswith('task_'):
            task_id = part.replace('task_', '')
        elif part.startswith('split_'):
            split_id = part.replace('split_', '')
    
    if not task_id:
        task_id = 'unknown'
    if not split_id:
        split_id = '0'
    
    # Process each model result
    for result in data:
        if not isinstance(result, dict):
            continue
        
        # Convert to standard format
        processed_result = {
            'model_name': result.get('model_name', 'unknown'),
            'dataset_name': result.get('dataset_name', 'unknown'), 
            'dataset_id': result.get('dataset_id', task_id),
            'task_id': task_id,
            'split_id': split_id,
            'task_type': result.get('task_type', 'classification'),
            'num_classes': result.get('num_classes', 2),
            'accuracy': result.get('accuracy'),
            'balanced_accuracy': result.get('balanced_accuracy'),
            'roc_auc': result.get('roc_auc'),
            '_archive_source': archive_name,
            '_file_source': file_name
        }
        
        processed_results.append(processed_result)
    
    return processed_results

def normalize_model_name(model_name: str) -> str:
    """Normalize model names for consistency."""
    model_name = model_name.lower().strip()
    
    # Map variations to standard names
    name_mapping = {
        'clam-t-sne-tabular': 'clam_tsne',
        'clam_t_sne_tabular': 'clam_tsne',
        'clam-tsne': 'clam_tsne',
        'clam-t-sne': 'clam_tsne',
        'clam_tsne': 'clam_tsne',
        'tabpfn_v2': 'tabpfn_v2',
        'tabpfnv2': 'tabpfn_v2'
    }
    
    return name_mapping.get(model_name, model_name)

def create_unique_model_identifier(model_name: str, archive_source: str) -> str:
    """Create a unique model identifier."""
    normalized_name = normalize_model_name(model_name)
    
    # Map clam_tsne to MARVIS for display
    if normalized_name == 'clam_tsne':
        return 'MARVIS'
    
    return normalized_name

def get_algorithm_scores_regression(results: List[Dict], algorithm_name: str) -> Dict[str, float]:
    """Extract regression scores for a specific algorithm."""
    dataset_scores = defaultdict(list)
    
    for entry in results:
        if not isinstance(entry, dict):
            continue
        
        # Handle different formats
        if entry.get('algorithm') == algorithm_name:
            dataset = entry.get('dataset_name', entry.get('task_id', 'unknown'))
            r2 = entry.get('r2_score', None)
        else:
            continue
            
        if r2 is not None and dataset != 'unknown':
            # Clamp negative R² to 0
            r2 = max(0.0, r2)
            dataset_scores[dataset].append(r2)
    
    # Average scores per dataset across splits
    dataset_avg_scores = {}
    for dataset, scores in dataset_scores.items():
        dataset_avg_scores[dataset] = statistics.mean(scores)
    
    return dataset_avg_scores

def get_algorithm_scores_classification(results: List[Dict], algorithm_name: str) -> Dict[str, float]:
    """Extract classification scores for a specific algorithm."""
    dataset_scores = defaultdict(list)
    
    for entry in results:
        if not isinstance(entry, dict):
            continue
        
        # Create unique model identifier
        original_model_name = entry.get('model_name', 'unknown')
        archive_source = entry.get('_archive_source', 'unknown')
        unique_model_name = create_unique_model_identifier(original_model_name, archive_source)
        
        if unique_model_name == algorithm_name:
            dataset = entry.get('dataset_name', entry.get('task_id', 'unknown'))
            # Use balanced accuracy for classification
            score = entry.get('balanced_accuracy', entry.get('accuracy', None))
        else:
            continue
            
        if score is not None and dataset != 'unknown':
            dataset_scores[dataset].append(score)
    
    # Average scores per dataset across splits
    dataset_avg_scores = {}
    for dataset, scores in dataset_scores.items():
        dataset_avg_scores[dataset] = statistics.mean(scores)
    
    return dataset_avg_scores

def analyze_correlation_regression(input_dir: str, output_dir: str):
    """Analyze correlation for regression tasks."""
    print("🔢 Analyzing REGRESSION correlation...")
    
    results_dir = Path(input_dir)
    
    tar_files = {
        'marvis': results_dir / "clam-reg.tar",
        'baselines': results_dir / "tabular_baselines_reg.tar"
    }
    
    all_results = []
    
    # Process each tar file
    for source, tar_path in tar_files.items():
        if not tar_path.exists():
            print(f"⚠️  Skipping {source}: {tar_path} not found")
            continue
            
        print(f"📦 Processing {source} from {tar_path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = extract_regression_results_from_tar(str(tar_path), temp_dir)
            all_results.extend(results)
            print(f"  Extracted {len(results)} results")
    
    # Load TabPFNv2 supplemental results
    supplemental_path = results_dir / "all_evaluation_results_20250626_133915.json"
    tabpfnv2_replacements = load_tabpfnv2_supplemental_results(str(supplemental_path))
    
    # Replace TabPFNv2 results with supplemental results where available
    tasks_to_replace = set(tabpfnv2_replacements.keys())
    original_count = len(all_results)
    
    # Remove existing TabPFNv2 results for tasks that have replacements
    all_results = [result for result in all_results 
                   if not (result['algorithm'] == 'tabpfn_v2' and result['task_id'] in tasks_to_replace)]
    
    # Add the supplemental TabPFNv2 results
    for task_id, replacement_result in tabpfnv2_replacements.items():
        all_results.append(replacement_result)
    
    print(f"📦 Replaced {original_count - len(all_results) + len(tabpfnv2_replacements)} TabPFNv2 results with supplemental data")
    
    # Extract scores for each algorithm
    marvis_scores = get_algorithm_scores_regression(all_results, 'MARVIS')
    tabpfn_scores = get_algorithm_scores_regression(all_results, 'tabpfn_v2')
    
    print(f"\nMARVIS datasets: {len(marvis_scores)}")
    print(f"TabPFN v2 datasets: {len(tabpfn_scores)}")
    
    # Find common datasets
    common_datasets = set(marvis_scores.keys()) & set(tabpfn_scores.keys())
    print(f"Common datasets: {len(common_datasets)}")
    
    if not common_datasets:
        print("No common datasets found for regression!")
        return
    
    # Create correlation plot
    create_correlation_plot(
        marvis_scores, tabpfn_scores, common_datasets,
        "MARVIS", "TabPFN v2", "R² Score", 
        "Regression", output_dir
    )

def analyze_correlation_classification(input_dir: str, output_dir: str):
    """Analyze correlation for classification tasks."""
    print("\n📊 Analyzing CLASSIFICATION correlation...")
    
    results_dir = Path(input_dir)
    
    # Classification tar files (excluding regression archives)
    tar_files = []
    regression_archives = {'jolt_reg.tar', 'clam-reg.tar', 'tabular_baselines_reg.tar'}
    
    for file_name in results_dir.glob("*.tar"):
        if file_name.name not in regression_archives:
            tar_files.append(file_name)
            print(f"Found classification archive: {file_name.name}")
    
    if not tar_files:
        print("⚠️  No classification tar files found!")
        return
    
    all_results = []
    
    # Process each tar file
    for tar_path in tar_files:
        print(f"📦 Processing {tar_path.name}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = extract_classification_results_from_tar(str(tar_path), temp_dir)
            all_results.extend(results)
            print(f"  Extracted {len(results)} results")
    
    # Extract scores for each algorithm
    marvis_scores = get_algorithm_scores_classification(all_results, 'MARVIS')
    tabpfn_scores = get_algorithm_scores_classification(all_results, 'tabpfn_v2')
    
    print(f"\nMARVIS datasets: {len(marvis_scores)}")
    print(f"TabPFN v2 datasets: {len(tabpfn_scores)}")
    
    # Find common datasets
    common_datasets = set(marvis_scores.keys()) & set(tabpfn_scores.keys())
    print(f"Common datasets: {len(common_datasets)}")
    
    if not common_datasets:
        print("No common datasets found for classification!")
        return
    
    # Create correlation plot
    create_correlation_plot(
        marvis_scores, tabpfn_scores, common_datasets,
        "MARVIS", "TabPFN v2", "Balanced Accuracy", 
        "Classification", output_dir
    )

def create_correlation_plot(algo1_scores: Dict[str, float], algo2_scores: Dict[str, float], 
                          common_datasets: set, algo1_name: str, algo2_name: str, 
                          metric_name: str, task_type: str, output_dir: str):
    """Create correlation plot for the given algorithms and datasets."""
    
    # Prepare paired scores
    algo1_values = []
    algo2_values = []
    dataset_names = []
    
    for dataset in sorted(common_datasets):
        algo1_values.append(algo1_scores[dataset])
        algo2_values.append(algo2_scores[dataset])
        dataset_names.append(dataset)
    
    algo1_array = np.array(algo1_values)
    algo2_array = np.array(algo2_values)
    
    # Calculate correlations
    correlation, p_value = stats.pearsonr(algo1_array, algo2_array)
    spearman_corr, spearman_p = stats.spearmanr(algo1_array, algo2_array)
    
    print(f"\n📊 {task_type} Pearson Correlation: {correlation:.4f} (p-value: {p_value:.4f})")
    print(f"📊 {task_type} Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
    
    # Performance thresholds based on task type
    if task_type.lower() == 'regression':
        high_threshold = 0.8
        low_threshold = 0.3
    else:  # classification
        high_threshold = 0.9
        low_threshold = 0.6
    
    # Find datasets where both performed well
    both_good = [(name, score1, score2) for name, score1, score2 in zip(dataset_names, algo1_values, algo2_values)
                 if score1 > high_threshold and score2 > high_threshold]
    
    print(f"\n✅ Datasets where BOTH performed well ({metric_name} > {high_threshold}): {len(both_good)}")
    for name, score1, score2 in sorted(both_good, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name:40s} {algo1_name}: {score1:.4f}, {algo2_name}: {score2:.4f}")
    
    # Find datasets where both performed poorly
    both_poor = [(name, score1, score2) for name, score1, score2 in zip(dataset_names, algo1_values, algo2_values)
                 if score1 < low_threshold and score2 < low_threshold]
    
    print(f"\n❌ Datasets where BOTH performed poorly ({metric_name} < {low_threshold}): {len(both_poor)}")
    for name, score1, score2 in sorted(both_poor, key=lambda x: x[1])[:5]:
        print(f"  {name:40s} {algo1_name}: {score1:.4f}, {algo2_name}: {score2:.4f}")
    
    # Find disagreements
    algo1_better = [(name, score1, score2, score1 - score2) for name, score1, score2 in zip(dataset_names, algo1_values, algo2_values)
                   if score1 - score2 > 0.2]
    
    print(f"\n🔵 Datasets where {algo1_name} >> {algo2_name} (diff > 0.2): {len(algo1_better)}")
    for name, score1, score2, diff in sorted(algo1_better, key=lambda x: x[3], reverse=True)[:5]:
        print(f"  {name:40s} {algo1_name}: {score1:.4f}, {algo2_name}: {score2:.4f} (diff: {diff:.4f})")
    
    algo2_better = [(name, score1, score2, score2 - score1) for name, score1, score2 in zip(dataset_names, algo1_values, algo2_values)
                     if score2 - score1 > 0.2]
    
    print(f"\n🟡 Datasets where {algo2_name} >> {algo1_name} (diff > 0.2): {len(algo2_better)}")
    for name, score1, score2, diff in sorted(algo2_better, key=lambda x: x[3], reverse=True)[:5]:
        print(f"  {name:40s} {algo1_name}: {score1:.4f}, {algo2_name}: {score2:.4f} (diff: {diff:.4f})")
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(algo1_array, algo2_array, alpha=0.6, s=50)
    
    # Add diagonal line
    min_val = min(min(algo1_array), min(algo2_array))
    max_val = max(max(algo1_array), max(algo2_array))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
    
    # Add regression line
    z = np.polyfit(algo1_array, algo2_array, 1)
    p = np.poly1d(z)
    plt.plot(algo1_array, p(algo1_array), "b-", alpha=0.8, 
             label=f'Regression: y={z[0]:.3f}x+{z[1]:.3f}')
    
    plt.xlabel(f'{algo1_name} {metric_name}', fontsize=12)
    plt.ylabel(f'{algo2_name} {metric_name}', fontsize=12)
    plt.title(f'{algo1_name} vs {algo2_name} Performance ({task_type})\nPearson r={correlation:.3f}, Spearman ρ={spearman_corr:.3f}', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some interesting points annotations
    for name, score1, score2 in both_good[:3]:
        if score1 > 0.95 and score2 > 0.95:
            plt.annotate(name, (score1, score2), fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # Create output directory and save plot
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f'marvis_tabpfn_correlation_{task_type.lower()}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📈 {task_type} scatter plot saved to '{output_file}'")
    
    plt.close()
    
    # Summary statistics
    print(f"\n📊 {task_type} Summary Statistics:")
    print(f"  Average |{algo1_name} - {algo2_name}|: {np.mean(np.abs(algo1_array - algo2_array)):.4f}")
    
    if task_type.lower() == 'regression':
        print(f"  Both > 0.5: {sum((algo1_array > 0.5) & (algo2_array > 0.5))} datasets")
        print(f"  Both < 0.5: {sum((algo1_array < 0.5) & (algo2_array < 0.5))} datasets")
        print(f"  Opposite performance (one >0.7, other <0.3): {sum(((algo1_array > 0.7) & (algo2_array < 0.3)) | ((algo1_array < 0.3) & (algo2_array > 0.7)))} datasets")
    else:  # classification
        print(f"  Both > 0.8: {sum((algo1_array > 0.8) & (algo2_array > 0.8))} datasets")
        print(f"  Both < 0.7: {sum((algo1_array < 0.7) & (algo2_array < 0.7))} datasets")
        print(f"  Opposite performance (one >0.9, other <0.6): {sum(((algo1_array > 0.9) & (algo2_array < 0.6)) | ((algo1_array < 0.6) & (algo2_array > 0.9)))} datasets")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze correlation between MARVIS and TabPFN v2 performance")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing tar archives with results"
    )
    parser.add_argument(
        "--results_dir", 
        type=str, 
        help="Directory containing tar archives with results (deprecated, use --input_dir)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./correlation_analysis_output",
        help="Directory to save correlation plots and analysis"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["regression", "classification", "both"],
        default="both",
        help="Type of tasks to analyze"
    )
    
    args = parser.parse_args()
    
    # Handle input_dir vs results_dir - prefer input_dir if provided
    if args.input_dir and args.results_dir:
        print("⚠️  Both --input_dir and --results_dir provided. Using --input_dir.")
        input_dir = args.input_dir
    elif args.input_dir:
        input_dir = args.input_dir
    elif args.results_dir:
        print("⚠️  --results_dir is deprecated. Please use --input_dir instead.")
        input_dir = args.results_dir
    else:
        # Default fallback
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        input_dir = str(project_root / "results")
        print(f"📁 Using default input directory: {input_dir}")
    
    print(f"🔍 Analyzing MARVIS vs TabPFN v2 correlation")
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📋 Task type: {args.task_type}")
    
    if args.task_type in ["regression", "both"]:
        analyze_correlation_regression(input_dir, args.output_dir)
    
    if args.task_type in ["classification", "both"]:
        analyze_correlation_classification(input_dir, args.output_dir)
    
    print(f"\n✅ Correlation analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()