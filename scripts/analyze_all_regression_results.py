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

def analyze_all_results(results: List[Dict]):
    """Analyze combined results from all sources."""
    
    # Group by algorithm
    algorithm_results = defaultdict(list)
    algorithm_dataset_results = defaultdict(lambda: defaultdict(list))
    algorithm_dataset_coverage = defaultdict(set)
    algorithm_task_split_results = defaultdict(list)  # For subset analysis
    
    for result in results:
        algorithm = result['algorithm']
        r2_score = result['r2_score']
        dataset = result['dataset_name']
        task_id = result.get('task_id', 'unknown')
        
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
    print("📊 DATASET COVERAGE ANALYSIS (out of 43 total regression datasets)")
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
        coverage_percent = (num_datasets / 43) * 100
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
    
    # Analyze all results
    if all_results:
        print(f"\n📊 Total results collected: {len(all_results)}")
        analyze_all_results(all_results)
    else:
        print("\n❌ No results found to analyze!")

if __name__ == "__main__":
    main()