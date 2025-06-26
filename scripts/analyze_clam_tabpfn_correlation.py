#!/usr/bin/env python3
"""
Analyze correlation between CLAM and TabPFN v2 performance on regression datasets.
"""

import json
import tarfile
import tempfile
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_and_load_results(tar_path: str) -> dict:
    """Extract tar file and load main results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(temp_dir)
        
        temp_path = Path(temp_dir)
        
        # Find main results file
        main_results = list(temp_path.glob('**/all_regression_results*.json'))
        if main_results:
            with open(main_results[0], 'r') as f:
                return json.load(f)
        
        # For baselines, aggregate from individual files
        results = []
        eval_files = list(temp_path.glob('**/all_evaluation_results*.json'))
        for eval_file in eval_files:
            with open(eval_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
        
        return results

def get_algorithm_scores(results, algorithm_name):
    """Extract scores for a specific algorithm."""
    dataset_scores = defaultdict(list)
    
    for entry in results:
        if isinstance(entry, dict):
            # Handle different formats
            if 'model_name' in entry and entry['model_name'] == algorithm_name:
                dataset = entry.get('dataset_name', 'unknown')
                r2 = entry.get('r2_score', entry.get('r2', None))
            elif 'algorithm' in entry and entry['algorithm'] == algorithm_name:
                dataset = entry.get('dataset_name', 'unknown')
                r2 = entry.get('r2_score', None)
            elif algorithm_name == 'CLAM' and 'r2_score' in entry:
                # Handle CLAM format which may not have algorithm field
                dataset = entry.get('dataset_name', 'unknown')
                r2 = entry.get('r2_score', None)
            else:
                continue
                
            if r2 is not None and dataset != 'unknown':
                # Clamp negative R² to 0
                r2 = max(0.0, r2)
                dataset_scores[dataset].append(r2)
    
    # Average scores per dataset
    dataset_avg_scores = {}
    for dataset, scores in dataset_scores.items():
        dataset_avg_scores[dataset] = np.mean(scores)
    
    return dataset_avg_scores

def analyze_correlation():
    """Main analysis function."""
    # Load results
    print("Loading results...")
    
    # Load CLAM results
    clam_results = extract_and_load_results('results/clam-reg.tar')
    clam_scores = get_algorithm_scores(clam_results, 'CLAM')
    
    # Load baseline results for TabPFN v2
    baseline_results = extract_and_load_results('results/tabular_baselines_reg.tar')
    tabpfn_scores = get_algorithm_scores(baseline_results, 'tabpfn_v2')
    
    print(f"\nCLAM datasets: {len(clam_scores)}")
    print(f"TabPFN v2 datasets: {len(tabpfn_scores)}")
    
    # Find common datasets
    common_datasets = set(clam_scores.keys()) & set(tabpfn_scores.keys())
    print(f"Common datasets: {len(common_datasets)}")
    
    if not common_datasets:
        print("No common datasets found!")
        return
    
    # Prepare paired scores
    clam_values = []
    tabpfn_values = []
    dataset_names = []
    
    for dataset in sorted(common_datasets):
        clam_values.append(clam_scores[dataset])
        tabpfn_values.append(tabpfn_scores[dataset])
        dataset_names.append(dataset)
    
    clam_array = np.array(clam_values)
    tabpfn_array = np.array(tabpfn_values)
    
    # Calculate correlation
    correlation, p_value = stats.pearsonr(clam_array, tabpfn_array)
    print(f"\n📊 Pearson Correlation: {correlation:.4f} (p-value: {p_value:.4f})")
    
    # Calculate Spearman correlation (rank-based)
    spearman_corr, spearman_p = stats.spearmanr(clam_array, tabpfn_array)
    print(f"📊 Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
    
    # Find datasets where both performed well (R² > 0.8)
    both_good = [(name, clam, tabpfn) for name, clam, tabpfn in zip(dataset_names, clam_values, tabpfn_values)
                 if clam > 0.8 and tabpfn > 0.8]
    
    print(f"\n✅ Datasets where BOTH performed well (R² > 0.8): {len(both_good)}")
    for name, clam, tabpfn in sorted(both_good, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {name:40s} CLAM: {clam:.4f}, TabPFN: {tabpfn:.4f}")
    
    # Find datasets where both performed poorly (R² < 0.3)
    both_poor = [(name, clam, tabpfn) for name, clam, tabpfn in zip(dataset_names, clam_values, tabpfn_values)
                 if clam < 0.3 and tabpfn < 0.3]
    
    print(f"\n❌ Datasets where BOTH performed poorly (R² < 0.3): {len(both_poor)}")
    for name, clam, tabpfn in sorted(both_poor, key=lambda x: x[1])[:10]:
        print(f"  {name:40s} CLAM: {clam:.4f}, TabPFN: {tabpfn:.4f}")
    
    # Find disagreements
    clam_better = [(name, clam, tabpfn, clam - tabpfn) for name, clam, tabpfn in zip(dataset_names, clam_values, tabpfn_values)
                   if clam - tabpfn > 0.2]
    
    print(f"\n🔵 Datasets where CLAM >> TabPFN (diff > 0.2): {len(clam_better)}")
    for name, clam, tabpfn, diff in sorted(clam_better, key=lambda x: x[3], reverse=True)[:5]:
        print(f"  {name:40s} CLAM: {clam:.4f}, TabPFN: {tabpfn:.4f} (diff: {diff:.4f})")
    
    tabpfn_better = [(name, clam, tabpfn, tabpfn - clam) for name, clam, tabpfn in zip(dataset_names, clam_values, tabpfn_values)
                     if tabpfn - clam > 0.2]
    
    print(f"\n🟡 Datasets where TabPFN >> CLAM (diff > 0.2): {len(tabpfn_better)}")
    for name, clam, tabpfn, diff in sorted(tabpfn_better, key=lambda x: x[3], reverse=True)[:5]:
        print(f"  {name:40s} CLAM: {clam:.4f}, TabPFN: {tabpfn:.4f} (diff: {diff:.4f})")
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(clam_array, tabpfn_array, alpha=0.6, s=50)
    
    # Add diagonal line
    min_val = min(min(clam_array), min(tabpfn_array))
    max_val = max(max(clam_array), max(tabpfn_array))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
    
    # Add regression line
    z = np.polyfit(clam_array, tabpfn_array, 1)
    p = np.poly1d(z)
    plt.plot(clam_array, p(clam_array), "b-", alpha=0.8, 
             label=f'Regression: y={z[0]:.3f}x+{z[1]:.3f}')
    
    plt.xlabel('CLAM R² Score', fontsize=12)
    plt.ylabel('TabPFN v2 R² Score', fontsize=12)
    plt.title(f'CLAM vs TabPFN v2 Performance\nPearson r={correlation:.3f}, Spearman ρ={spearman_corr:.3f}', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some interesting points
    for name, clam, tabpfn in both_good[:3]:
        if clam > 0.95 and tabpfn > 0.95:
            plt.annotate(name, (clam, tabpfn), fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('clam_tabpfn_correlation.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 Scatter plot saved to 'clam_tabpfn_correlation.png'")
    
    # Summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"  Average |CLAM - TabPFN|: {np.mean(np.abs(clam_array - tabpfn_array)):.4f}")
    print(f"  Both > 0.5: {sum((clam_array > 0.5) & (tabpfn_array > 0.5))} datasets")
    print(f"  Both < 0.5: {sum((clam_array < 0.5) & (tabpfn_array < 0.5))} datasets")
    print(f"  Opposite performance (one >0.7, other <0.3): {sum(((clam_array > 0.7) & (tabpfn_array < 0.3)) | ((clam_array < 0.3) & (tabpfn_array > 0.7)))} datasets")

if __name__ == "__main__":
    analyze_correlation()