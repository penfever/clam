#!/usr/bin/env python3
"""
Analyze regression results from multiple tar files and JSON formats.

This script reads regression results from:
- results/clam-reg.tar (CLAM results)  
- results/jolt_reg.tar (JOLT results)
- results/tabular_baselines_reg.tar (Tabular baseline results)

It handles different JSON structures and provides a unified comparison.

Usage:
    python scripts/analyze_all_regression_results.py
"""

import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from collections import defaultdict
import statistics
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def extract_tar_and_find_results(tar_path: str, temp_dir: str) -> List[str]:
    """
    Extract tar file and find all result JSON files.
    
    Args:
        tar_path: Path to tar file
        temp_dir: Temporary directory to extract to
        
    Returns:
        List of paths to result JSON files
    """
    result_files = []
    
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(temp_dir)
    
    temp_path = Path(temp_dir)
    
    # First priority: Look for main aggregated results files
    main_results = list(temp_path.glob('**/all_regression_results*.json'))
    if main_results:
        # If we have a main results file, use only that to avoid duplicates
        result_files.extend(main_results)
        return [str(f) for f in result_files]
    
    # Second priority: For baselines, look for evaluation results
    eval_results = list(temp_path.glob('**/all_evaluation_results*.json'))
    if eval_results:
        # For baselines, each all_evaluation_results file contains results for one task/split
        # So we should include all of them
        result_files.extend(eval_results)
        return [str(f) for f in result_files]
    
    # Third priority: If no main files, look for individual result files
    # but track task/split to avoid duplicates
    processed_task_splits = set()
    
    # Process individual files, ensuring one per task/split
    for pattern in ['**/jolt_results.json', '**/aggregated_results.json']:
        for file_path in temp_path.glob(pattern):
            # Extract task and split from path
            parts = file_path.parts
            task_id = None
            split_id = None
            
            for i, part in enumerate(parts):
                if part.startswith('task_'):
                    task_id = part
                elif part.startswith('split_'):
                    split_id = part
            
            if task_id and split_id:
                key = f"{task_id}/{split_id}"
                if key not in processed_task_splits:
                    processed_task_splits.add(key)
                    result_files.append(file_path)
    
    return [str(f) for f in result_files]

def parse_clam_format(data: List[Dict]) -> List[Dict]:
    """Parse CLAM format results."""
    results = []
    for entry in data:
        results.append({
            'algorithm': 'CLAM',
            'dataset_name': entry.get('dataset_name', 'unknown'),
            'task_id': entry.get('task_id', 'unknown'),
            'r2_score': entry.get('r2_score', None),
            'mae': entry.get('mae', None),
            'rmse': entry.get('rmse', None)
        })
    return results

def parse_jolt_format(file_path: str, data: Dict) -> List[Dict]:
    """Parse JOLT format results."""
    results = []
    
    # JOLT results might be nested in different ways
    if 'results' in data:
        # Format 1: {results: {dataset: {metric: value}}}
        for dataset, metrics in data['results'].items():
            r2_score = metrics.get('r2_score', metrics.get('r2', None))
            results.append({
                'algorithm': 'JOLT',
                'dataset_name': dataset,
                'task_id': 'unknown',
                'r2_score': r2_score,
                'mae': metrics.get('mae', None),
                'rmse': metrics.get('rmse', None)
            })
    elif isinstance(data, list):
        # Format 2: List of results
        for entry in data:
            if 'model' in entry:
                algorithm = entry['model']
            else:
                algorithm = 'JOLT'
                
            results.append({
                'algorithm': algorithm,
                'dataset_name': entry.get('dataset', entry.get('dataset_name', 'unknown')),
                'task_id': entry.get('task_id', 'unknown'),
                'r2_score': entry.get('r2_score', entry.get('r2', None)),
                'mae': entry.get('mae', None),
                'rmse': entry.get('rmse', None)
            })
    else:
        # Format 3: Direct metrics
        # Try to extract dataset name from file path
        path_parts = Path(file_path).parts
        dataset_name = 'unknown'
        for i, part in enumerate(path_parts):
            if part.startswith('dataset_'):
                dataset_name = part.replace('dataset_', '')
            elif part.startswith('task_'):
                dataset_name = part
                
        r2_score = data.get('r2_score', data.get('r2', data.get('test_r2', None)))
        results.append({
            'algorithm': 'JOLT',
            'dataset_name': dataset_name,
            'task_id': 'unknown',
            'r2_score': r2_score,
            'mae': data.get('mae', data.get('test_mae', None)),
            'rmse': data.get('rmse', data.get('test_rmse', None))
        })
    
    return results

def parse_baseline_format(file_path: str, data: any) -> List[Dict]:
    """Parse tabular baseline format results."""
    results = []
    
    if isinstance(data, dict):
        # Check if it's an evaluation summary
        if 'model_results' in data:
            # Format: {model_results: {model_name: {metrics}}}
            for model_name, metrics in data['model_results'].items():
                r2_score = metrics.get('r2_score', metrics.get('r2', metrics.get('test_r2', None)))
                results.append({
                    'algorithm': model_name,
                    'dataset_name': data.get('dataset_name', 'unknown'),
                    'task_id': data.get('task_id', 'unknown'),
                    'r2_score': r2_score,
                    'mae': metrics.get('mae', metrics.get('test_mae', None)),
                    'rmse': metrics.get('rmse', metrics.get('test_rmse', None))
                })
        else:
            # Try to handle as single model result
            for key, value in data.items():
                if isinstance(value, dict) and any(k in value for k in ['r2', 'r2_score', 'test_r2']):
                    r2_score = value.get('r2_score', value.get('r2', value.get('test_r2', None)))
                    results.append({
                        'algorithm': key,
                        'dataset_name': 'unknown',
                        'task_id': 'unknown', 
                        'r2_score': r2_score,
                        'mae': value.get('mae', value.get('test_mae', None)),
                        'rmse': value.get('rmse', value.get('test_rmse', None))
                    })
    elif isinstance(data, list):
        # List format - most common for baseline results
        for entry in data:
            # Extract the specific model name (e.g., 'catboost', 'random_forest')
            model_name = entry.get('model_name', entry.get('algorithm', entry.get('model', 'baseline')))
            dataset_name = entry.get('dataset_name', entry.get('dataset', 'unknown'))
            
            # Extract R² score from various possible locations
            r2_score = None
            if 'metrics' in entry:
                r2_score = entry['metrics'].get('r2_score', entry['metrics'].get('r2', entry['metrics'].get('test_r2', None)))
            else:
                r2_score = entry.get('r2_score', entry.get('r2', entry.get('test_r2', None)))
            
            results.append({
                'algorithm': model_name,  # Use specific model name
                'dataset_name': dataset_name,
                'task_id': entry.get('task_id', entry.get('dataset_id', 'unknown')),
                'r2_score': r2_score,
                'mae': entry.get('mae', entry.get('test_mae', None)),
                'rmse': entry.get('rmse', entry.get('test_rmse', None))
            })
    
    return results

def process_json_file(file_path: str, source: str) -> List[Dict]:
    """Process a single JSON file and extract results."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading {file_path}: {e}")
        return []
    
    # Determine format and parse accordingly
    if source == 'clam' and isinstance(data, list):
        return parse_clam_format(data)
    elif source == 'jolt' or 'jolt' in file_path.lower():
        return parse_jolt_format(file_path, data)
    else:
        return parse_baseline_format(file_path, data)

def create_critical_difference_plot(algorithm_scores_matrix: Dict[str, Dict[str, float]], 
                                  output_path: str, 
                                  title: str = "Critical Difference Diagram",
                                  alpha: float = 0.05):
    """
    Create a critical difference plot for algorithm comparison.
    
    Args:
        algorithm_scores_matrix: Dict mapping algorithm -> dataset -> score
        output_path: Path to save the plot
        title: Title for the plot
        alpha: Significance level for statistical tests
    """
    try:
        import scikit_posthocs as sp
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("⚠️  scikit-posthocs not installed. Installing...")
        os.system("pip install scikit-posthocs")
        import scikit_posthocs as sp
        from matplotlib.backends.backend_pdf import PdfPages
    
    # Convert to pandas DataFrame
    algorithms = list(algorithm_scores_matrix.keys())
    datasets = set()
    for alg_data in algorithm_scores_matrix.values():
        datasets.update(alg_data.keys())
    datasets = sorted(list(datasets))
    
    # Create matrix with algorithms as columns and datasets as rows
    data_matrix = []
    for dataset in datasets:
        row = []
        for algorithm in algorithms:
            # Use the score if available, otherwise use NaN
            score = algorithm_scores_matrix[algorithm].get(dataset, np.nan)
            row.append(score)
        data_matrix.append(row)
    
    df = pd.DataFrame(data_matrix, columns=algorithms, index=datasets)
    
    # Remove rows with any NaN values for fair comparison
    df_clean = df.dropna()
    
    if len(df_clean) < 3:
        print(f"⚠️  Not enough complete datasets ({len(df_clean)}) for statistical testing. Skipping CD plot.")
        return
    
    # Perform Friedman test
    stat, p_value = stats.friedmanchisquare(*[df_clean[col] for col in df_clean.columns])
    
    print(f"\n📊 Friedman Test Results:")
    print(f"   Statistic: {stat:.4f}")
    print(f"   p-value: {p_value:.4f}")
    
    if p_value < alpha:
        print(f"   ✅ Significant differences found (p < {alpha})")
        
        # Perform post-hoc Nemenyi test
        nemenyi_results = sp.posthoc_nemenyi_friedman(df_clean.values)
        
        # Calculate average ranks
        ranks = df_clean.rank(axis=1, ascending=False, method='average')
        avg_ranks = ranks.mean(axis=0).sort_values()
        
        print(f"\n📊 Average Ranks:")
        for i, (alg, rank) in enumerate(avg_ranks.items(), 1):
            print(f"   {i}. {alg}: {rank:.3f}")
        
        # Create critical difference plot
        plt.figure(figsize=(10, 6))
        
        # Use scikit-posthocs CD diagram
        sp.critical_difference_diagram(
            avg_ranks.to_dict(),
            nemenyi_results,
            label_fmt_left='{label} ({rank:.2f})',
            label_fmt_right='{label} ({rank:.2f})',
            label_props={'size': 12},
            color_palette=['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        )
        
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Critical difference plot saved to: {output_path}")
    else:
        print(f"   ❌ No significant differences found (p >= {alpha})")
        
def create_performance_matrix_plot(algorithm_dataset_results: Dict[str, Dict[str, List[float]]], 
                                 output_path: str,
                                 metric_name: str = "R²"):
    """
    Create a heatmap showing algorithm performance across datasets.
    
    Args:
        algorithm_dataset_results: Dict mapping algorithm -> dataset -> [scores]
        output_path: Path to save the plot
        metric_name: Name of the metric being plotted
    """
    # Get all algorithms and datasets
    algorithms = sorted(algorithm_dataset_results.keys())
    datasets = set()
    for alg_data in algorithm_dataset_results.values():
        datasets.update(alg_data.keys())
    datasets = sorted(list(datasets))
    
    # Create matrix
    matrix = np.zeros((len(algorithms), len(datasets)))
    
    for i, algorithm in enumerate(algorithms):
        for j, dataset in enumerate(datasets):
            if dataset in algorithm_dataset_results[algorithm]:
                scores = algorithm_dataset_results[algorithm][dataset]
                matrix[i, j] = statistics.mean(scores) if scores else 0
            else:
                matrix[i, j] = np.nan
    
    # Create heatmap
    plt.figure(figsize=(20, 8))
    
    # Use masked array to handle NaN values
    masked_matrix = np.ma.masked_invalid(matrix)
    
    im = plt.imshow(masked_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    plt.xticks(range(len(datasets)), datasets, rotation=90, ha='right')
    plt.yticks(range(len(algorithms)), algorithms)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label(f'Average {metric_name}', rotation=270, labelpad=20)
    
    # Add text annotations for values
    for i in range(len(algorithms)):
        for j in range(len(datasets)):
            if not np.isnan(matrix[i, j]):
                text = plt.text(j, i, f'{matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=6)
    
    plt.title(f'Algorithm Performance Matrix ({metric_name})', fontsize=14, pad=20)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Algorithm', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Performance matrix plot saved to: {output_path}")

def analyze_all_results(results: List[Dict]):
    """Analyze combined results from all sources."""
    
    # Group by algorithm
    algorithm_results = defaultdict(list)
    algorithm_dataset_results = defaultdict(lambda: defaultdict(list))
    algorithm_dataset_coverage = defaultdict(set)
    algorithm_task_split_results = defaultdict(list)  # For subset analysis
    
    # First pass: collect all results
    all_dataset_results = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        algorithm = result['algorithm']
        r2_score = result['r2_score']
        dataset = result['dataset_name']
        task_id = result.get('task_id', 'unknown')
        
        if dataset != 'unknown' and r2_score is not None:
            # Apply minimum of 0
            if r2_score < 0:
                r2_score = 0.0
            all_dataset_results[dataset][algorithm].append(r2_score)
    
    # Filter out datasets where all algorithms perform poorly
    valid_datasets = set()
    filtered_dataset_count = 0
    
    print("\n" + "="*80)
    print("📊 DATASET FILTERING")
    print("="*80)
    
    for dataset, algorithm_scores in all_dataset_results.items():
        max_avg_score = 0.0
        for algorithm, scores in algorithm_scores.items():
            if scores:
                # Use average across splits, not max of individual splits
                avg_score = statistics.mean(scores)
                max_avg_score = max(max_avg_score, avg_score)
        
        if max_avg_score >= 0.1:
            valid_datasets.add(dataset)
        else:
            filtered_dataset_count += 1
            print(f"  Filtering out '{dataset}' - max average R² = {max_avg_score:.6f}")
    
    print(f"\n  Total datasets filtered out: {filtered_dataset_count}")
    print(f"  Remaining valid datasets: {len(valid_datasets)}")
    
    # Second pass: only include results from valid datasets
    for result in results:
        algorithm = result['algorithm']
        r2_score = result['r2_score']
        dataset = result['dataset_name']
        task_id = result.get('task_id', 'unknown')
        
        # Only process if dataset is valid
        if dataset not in valid_datasets:
            continue
            
        # Track dataset coverage
        if dataset != 'unknown':
            algorithm_dataset_coverage[algorithm].add(dataset)
        
        if r2_score is not None:
            # Apply minimum of 0
            if r2_score < 0:
                r2_score = 0.0
            
            algorithm_results[algorithm].append(r2_score)
            algorithm_dataset_results[algorithm][dataset].append(r2_score)
            
            # Store with task/dataset info for subset analysis
            algorithm_task_split_results[algorithm].append({
                'r2_score': r2_score,
                'dataset': dataset,
                'task_id': task_id
            })
    
    # Print overall comparison
    print("\n" + "="*80)
    print("📊 ALGORITHM COMPARISON - OVERALL R² SCORES")
    print("="*80)
    
    algorithm_stats = []
    for algorithm in sorted(algorithm_results.keys()):
        scores = algorithm_results[algorithm]
        if scores:
            avg_r2 = statistics.mean(scores)
            median_r2 = statistics.median(scores)
            min_r2 = min(scores)
            max_r2 = max(scores)
            n_samples = len(scores)
            
            algorithm_stats.append((avg_r2, algorithm))
            
            print(f"\n{algorithm:20s}")
            print(f"  Samples:     {n_samples}")
            print(f"  Average R²:  {avg_r2:.6f}")
            print(f"  Median R²:   {median_r2:.6f}")
            print(f"  Min R²:      {min_r2:.6f}")
            print(f"  Max R²:      {max_r2:.6f}")
    
    # Rank algorithms
    print("\n" + "="*80)
    print("🏆 ALGORITHM RANKING BY AVERAGE R²")
    print("="*80)
    
    algorithm_stats.sort(reverse=True)
    for rank, (avg_r2, algorithm) in enumerate(algorithm_stats, 1):
        print(f"{rank}. {algorithm:20s}: {avg_r2:.6f}")
    
    # Dataset-level comparison
    print("\n" + "="*80)
    print("📋 DATASET-LEVEL COMPARISON")
    print("="*80)
    
    # Get all unique datasets
    all_datasets = set()
    for algorithm in algorithm_dataset_results:
        all_datasets.update(algorithm_dataset_results[algorithm].keys())
    
    # For each dataset, show algorithm comparison
    for dataset in sorted(all_datasets):
        if dataset == 'unknown':
            continue
            
        print(f"\n{dataset}:")
        dataset_scores = []
        
        for algorithm in sorted(algorithm_dataset_results.keys()):
            if dataset in algorithm_dataset_results[algorithm]:
                scores = algorithm_dataset_results[algorithm][dataset]
                avg_score = statistics.mean(scores)
                dataset_scores.append((avg_score, algorithm))
        
        # Sort by score
        dataset_scores.sort(reverse=True)
        for score, algorithm in dataset_scores:
            print(f"  {algorithm:20s}: {score:.6f}")
    
    # Performance distribution
    print("\n" + "="*80)
    print("📊 PERFORMANCE DISTRIBUTION BY ALGORITHM")
    print("="*80)
    
    for algorithm in sorted(algorithm_results.keys()):
        scores = algorithm_results[algorithm]
        if scores:
            excellent = sum(1 for r2 in scores if r2 >= 0.9)
            good = sum(1 for r2 in scores if 0.7 <= r2 < 0.9)
            fair = sum(1 for r2 in scores if 0.5 <= r2 < 0.7)
            poor = sum(1 for r2 in scores if 0.0 <= r2 < 0.5)
            
            total = len(scores)
            
            print(f"\n{algorithm}:")
            print(f"  Excellent (R² ≥ 0.9):  {excellent:3d} ({excellent/total*100:5.1f}%)")
            print(f"  Good (0.7 ≤ R² < 0.9): {good:3d} ({good/total*100:5.1f}%)")
            print(f"  Fair (0.5 ≤ R² < 0.7): {fair:3d} ({fair/total*100:5.1f}%)")
            print(f"  Poor (0.0 ≤ R² < 0.5): {poor:3d} ({poor/total*100:5.1f}%)")
    
    # Dataset coverage analysis
    print("\n" + "="*80)
    print(f"📊 DATASET COVERAGE ANALYSIS (out of {len(valid_datasets)} valid datasets after filtering)")
    print("="*80)
    
    # Get all unique datasets across all algorithms
    all_covered_datasets = set()
    for algorithm in algorithm_dataset_coverage:
        all_covered_datasets.update(algorithm_dataset_coverage[algorithm])
    
    print(f"\nTotal unique datasets covered across all algorithms: {len(all_covered_datasets)}")
    
    # Sort algorithms by coverage
    coverage_stats = []
    for algorithm in algorithm_dataset_coverage:
        num_datasets = len(algorithm_dataset_coverage[algorithm])
        coverage_stats.append((num_datasets, algorithm))
    
    coverage_stats.sort(reverse=True)
    
    print("\nDataset coverage by algorithm:")
    for num_datasets, algorithm in coverage_stats:
        coverage_percent = (num_datasets / len(valid_datasets)) * 100
        print(f"  {algorithm:25s}: {num_datasets:2d} datasets ({coverage_percent:5.1f}%)")
    
    # Find datasets not covered by any algorithm
    if len(all_covered_datasets) < 43:
        print(f"\n⚠️  {43 - len(all_covered_datasets)} datasets were not successfully evaluated by any algorithm")
    
    # Subset analysis: Compare only on datasets where TabPFN v2 ran successfully
    print("\n" + "="*80)
    print("📊 SUBSET ANALYSIS: Comparison on TabPFN v2 Successful Datasets Only")
    print("="*80)
    
    if 'tabpfn_v2' not in algorithm_task_split_results:
        print("TabPFN v2 results not found for subset analysis")
        return
    
    # Get datasets where TabPFN v2 ran successfully
    tabpfn_datasets = set()
    for entry in algorithm_task_split_results['tabpfn_v2']:
        tabpfn_datasets.add(entry['dataset'])
    
    print(f"TabPFN v2 ran successfully on {len(tabpfn_datasets)} datasets")
    print(f"This represents {len(algorithm_task_split_results['tabpfn_v2'])} dataset splits")
    
    # Filter results to only include TabPFN datasets
    subset_algorithm_results = defaultdict(list)
    subset_algorithm_coverage = defaultdict(set)
    
    for algorithm, entries in algorithm_task_split_results.items():
        for entry in entries:
            if entry['dataset'] in tabpfn_datasets:
                subset_algorithm_results[algorithm].append(entry['r2_score'])
                subset_algorithm_coverage[algorithm].add(entry['dataset'])
    
    print(f"\n📈 SUBSET PERFORMANCE COMPARISON (TabPFN datasets only)")
    print("="*60)
    
    subset_stats = []
    for algorithm in sorted(subset_algorithm_results.keys()):
        scores = subset_algorithm_results[algorithm]
        if scores:
            avg_r2 = statistics.mean(scores)
            median_r2 = statistics.median(scores)
            n_samples = len(scores)
            n_datasets = len(subset_algorithm_coverage[algorithm])
            
            subset_stats.append((avg_r2, algorithm))
            
            print(f"\n{algorithm:20s}")
            print(f"  Samples:       {n_samples}")
            print(f"  Datasets:      {n_datasets}")
            print(f"  Average R²:    {avg_r2:.6f}")
            print(f"  Median R²:     {median_r2:.6f}")
    
    # Subset ranking
    print(f"\n🏆 SUBSET RANKING BY AVERAGE R² (TabPFN datasets only)")
    print("="*60)
    
    subset_stats.sort(reverse=True)
    for rank, (avg_r2, algorithm) in enumerate(subset_stats, 1):
        # Calculate coverage on TabPFN datasets
        coverage = len(subset_algorithm_coverage[algorithm])
        coverage_pct = (coverage / len(tabpfn_datasets)) * 100
        print(f"{rank}. {algorithm:20s}: {avg_r2:.6f} ({coverage}/{len(tabpfn_datasets)} datasets, {coverage_pct:.1f}%)")
    
    # Performance distribution for subset
    print(f"\n📊 SUBSET PERFORMANCE DISTRIBUTION")
    print("="*60)
    
    for algorithm in sorted(subset_algorithm_results.keys()):
        scores = subset_algorithm_results[algorithm]
        if scores:
            excellent = sum(1 for r2 in scores if r2 >= 0.9)
            good = sum(1 for r2 in scores if 0.7 <= r2 < 0.9)
            fair = sum(1 for r2 in scores if 0.5 <= r2 < 0.7)
            poor = sum(1 for r2 in scores if 0.0 <= r2 < 0.5)
            
            total = len(scores)
            
            print(f"\n{algorithm}:")
            print(f"  Excellent (R² ≥ 0.9):  {excellent:3d} ({excellent/total*100:5.1f}%)")
            print(f"  Good (0.7 ≤ R² < 0.9): {good:3d} ({good/total*100:5.1f}%)")
            print(f"  Fair (0.5 ≤ R² < 0.7): {fair:3d} ({fair/total*100:5.1f}%)")
            print(f"  Poor (0.0 ≤ R² < 0.5): {poor:3d} ({poor/total*100:5.1f}%)")
    
    # Win rate analysis
    print("\n" + "="*80)
    print("🏆 WIN RATE ANALYSIS")
    print("="*80)
    
    # Calculate per-dataset wins
    dataset_wins = defaultdict(lambda: defaultdict(int))  # dataset -> algorithm -> wins
    dataset_algorithm_results = defaultdict(lambda: defaultdict(list))  # dataset -> algorithm -> [scores]
    
    # Group all results by dataset and algorithm
    for result in results:
        algorithm = result['algorithm']
        dataset = result['dataset_name']
        r2_score = result['r2_score']
        
        if r2_score is not None and dataset != 'unknown':
            # Apply minimum of 0
            if r2_score < 0:
                r2_score = 0.0
            dataset_algorithm_results[dataset][algorithm].append(r2_score)
    
    # For each dataset, find the best average performer
    total_datasets_evaluated = 0
    algorithm_wins = defaultdict(int)
    algorithm_dataset_participation = defaultdict(int)
    
    for dataset in sorted(dataset_algorithm_results.keys()):
        # Only process valid datasets
        if dataset not in valid_datasets:
            continue
            
        # Calculate average performance for each algorithm on this dataset
        dataset_avg_scores = {}
        for algorithm, scores in dataset_algorithm_results[dataset].items():
            if scores:  # Only consider algorithms that have results
                avg_score = statistics.mean(scores)
                dataset_avg_scores[algorithm] = avg_score
                algorithm_dataset_participation[algorithm] += 1
        
        if dataset_avg_scores:
            total_datasets_evaluated += 1
            # Find the winner (highest average R²)
            winner = max(dataset_avg_scores.items(), key=lambda x: x[1])
            winner_algorithm = winner[0]
            winner_score = winner[1]
            
            algorithm_wins[winner_algorithm] += 1
            
            # Show dataset result
            print(f"\n{dataset}:")
            print(f"  Winner: {winner_algorithm} (R² = {winner_score:.6f})")
            
            # Show top 3 performers
            sorted_performers = sorted(dataset_avg_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (alg, score) in enumerate(sorted_performers[:3], 1):
                print(f"  {rank}. {alg:20s}: {score:.6f}")
    
    # Calculate win rates
    print(f"\n🏆 OVERALL WIN RATES (out of {total_datasets_evaluated} datasets)")
    print("="*60)
    
    win_rate_stats = []
    for algorithm in sorted(algorithm_wins.keys()):
        wins = algorithm_wins[algorithm]
        participation = algorithm_dataset_participation[algorithm]
        win_rate = (wins / total_datasets_evaluated) * 100
        participation_rate = (participation / total_datasets_evaluated) * 100
        
        win_rate_stats.append((wins, win_rate, algorithm))
        
        print(f"{algorithm:25s}: {wins:2d} wins ({win_rate:5.1f}%) | Participated in {participation} datasets ({participation_rate:5.1f}%)")
    
    # Sort by wins
    win_rate_stats.sort(reverse=True)
    
    print(f"\n🥇 WIN RATE RANKING")
    print("="*40)
    for rank, (wins, win_rate, algorithm) in enumerate(win_rate_stats, 1):
        print(f"{rank}. {algorithm:20s}: {wins:2d} wins ({win_rate:5.1f}%)")
    
    # Success rate analysis (R² ≥ 0.5 threshold)
    print(f"\n📈 SUCCESS RATE ANALYSIS (R² ≥ 0.5 per dataset)")
    print("="*60)
    
    algorithm_success_stats = defaultdict(lambda: {'successes': 0, 'total_datasets': 0})
    
    for dataset in dataset_algorithm_results:
        # Only process valid datasets
        if dataset not in valid_datasets:
            continue
            
        for algorithm, scores in dataset_algorithm_results[dataset].items():
            if scores:  # Algorithm participated in this dataset
                avg_score = statistics.mean(scores)
                algorithm_success_stats[algorithm]['total_datasets'] += 1
                if avg_score >= 0.5:
                    algorithm_success_stats[algorithm]['successes'] += 1
    
    success_rate_stats = []
    for algorithm in sorted(algorithm_success_stats.keys()):
        stats = algorithm_success_stats[algorithm]
        successes = stats['successes']
        total = stats['total_datasets']
        success_rate = (successes / total) * 100 if total > 0 else 0
        
        success_rate_stats.append((success_rate, successes, total, algorithm))
        
        print(f"{algorithm:25s}: {successes:2d}/{total:2d} datasets successful ({success_rate:5.1f}%)")
    
    # Sort by success rate
    success_rate_stats.sort(reverse=True)
    
    print(f"\n📊 SUCCESS RATE RANKING (R² ≥ 0.5)")
    print("="*50)
    for rank, (success_rate, successes, total, algorithm) in enumerate(success_rate_stats, 1):
        print(f"{rank}. {algorithm:20s}: {successes:2d}/{total:2d} ({success_rate:5.1f}%)")
    
    # Generate plots
    print("\n" + "="*80)
    print("📊 GENERATING STATISTICAL PLOTS")
    print("="*80)
    
    # Create output directory for plots
    script_dir = Path(__file__).parent
    plots_dir = script_dir / "regression_analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Prepare data for critical difference plot
    algorithm_scores_matrix = {}
    for algorithm in algorithm_dataset_results:
        algorithm_scores_matrix[algorithm] = {}
        for dataset in dataset_algorithm_results:
            # Only include valid datasets
            if dataset not in valid_datasets:
                continue
            if algorithm in dataset_algorithm_results[dataset]:
                scores = dataset_algorithm_results[dataset][algorithm]
                if scores:
                    avg_score = statistics.mean(scores)
                    algorithm_scores_matrix[algorithm][dataset] = avg_score
    
    # Create critical difference plot
    cd_plot_path = plots_dir / "critical_difference_r2.png"
    create_critical_difference_plot(
        algorithm_scores_matrix,
        str(cd_plot_path),
        title="Critical Difference Diagram - R² Performance",
        alpha=0.05
    )
    
    # Create performance matrix heatmap
    matrix_plot_path = plots_dir / "performance_matrix_heatmap.png"
    create_performance_matrix_plot(
        algorithm_dataset_results,
        str(matrix_plot_path),
        metric_name="R²"
    )
    
    # Create separate CD plot for success rates (binary: success or not)
    algorithm_success_matrix = {}
    for algorithm in algorithm_dataset_results:
        algorithm_success_matrix[algorithm] = {}
        for dataset in dataset_algorithm_results:
            # Only include valid datasets
            if dataset not in valid_datasets:
                continue
            if algorithm in dataset_algorithm_results[dataset]:
                scores = dataset_algorithm_results[dataset][algorithm]
                if scores:
                    avg_score = statistics.mean(scores)
                    # Binary success: 1 if R² >= 0.5, 0 otherwise
                    algorithm_success_matrix[algorithm][dataset] = 1.0 if avg_score >= 0.5 else 0.0
    
    cd_success_plot_path = plots_dir / "critical_difference_success_rate.png"
    create_critical_difference_plot(
        algorithm_success_matrix,
        str(cd_success_plot_path),
        title="Critical Difference Diagram - Success Rate (R² ≥ 0.5)",
        alpha=0.05
    )
    
    print(f"\n✅ All plots saved to: {plots_dir}")

def main():
    """Main function."""
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / "results"
    
    tar_files = {
        'clam': results_dir / "clam-reg.tar",
        'jolt': results_dir / "jolt_reg.tar", 
        'baselines': results_dir / "tabular_baselines_reg.tar"
    }
    
    all_results = []
    
    # Process each tar file
    for source, tar_path in tar_files.items():
        if not tar_path.exists():
            print(f"⚠️  Skipping {source}: {tar_path} not found")
            continue
            
        print(f"\n📦 Processing {source} from {tar_path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract and find result files
            result_files = extract_tar_and_find_results(str(tar_path), temp_dir)
            print(f"  Found {len(result_files)} result files")
            
            # Process each JSON file
            for json_file in result_files:
                print(f"  Processing: {Path(json_file).name}")
                results = process_json_file(json_file, source)
                all_results.extend(results)
                print(f"    Extracted {len(results)} results")
    
    # Process supplemental TabPFN v2 results
    supplemental_file = results_dir / "all_evaluation_results_20250626_133915.json"
    if supplemental_file.exists():
        print(f"\n📦 Processing supplemental TabPFN v2 results from {supplemental_file.name}")
        
        # Get new TabPFN v2 results
        supplemental_results = process_json_file(str(supplemental_file), 'baselines')
        new_tabpfn_results = [r for r in supplemental_results if r['algorithm'] == 'tabpfn_v2']
        
        # Create a set of dataset names from the new results
        new_dataset_names = {r['dataset_name'] for r in new_tabpfn_results}
        
        # Remove existing TabPFN v2 results ONLY for datasets that have new results
        original_count = len(all_results)
        all_results = [r for r in all_results if not (r['algorithm'] == 'tabpfn_v2' and r['dataset_name'] in new_dataset_names)]
        removed_count = original_count - len(all_results)
        
        if removed_count > 0:
            print(f"  Replaced {removed_count} existing TabPFN v2 results for datasets: {sorted(new_dataset_names)}")
        
        # Add new TabPFN v2 results
        all_results.extend(new_tabpfn_results)
        print(f"  Added {len(new_tabpfn_results)} new TabPFN v2 results")
        
        # Count total TabPFN v2 results after merge
        total_tabpfn = len([r for r in all_results if r['algorithm'] == 'tabpfn_v2'])
        print(f"  Total TabPFN v2 results after merge: {total_tabpfn}")
    else:
        print(f"\n⚠️  Supplemental file not found: {supplemental_file}")
    
    # Analyze all results
    if all_results:
        print(f"\n📊 Total results collected: {len(all_results)}")
        analyze_all_results(all_results)
    else:
        print("\n❌ No results found to analyze!")

if __name__ == "__main__":
    main()