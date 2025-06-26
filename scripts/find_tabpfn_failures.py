#!/usr/bin/env python3
"""
Find datasets where TabPFN v2 failed to run.
"""

import json
import tarfile
import tempfile
from pathlib import Path
from collections import defaultdict

def get_all_datasets_from_clam():
    """Get all datasets from CLAM results (which covers all 43 datasets)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open('results/clam-reg.tar', 'r') as tar:
            tar.extractall(temp_dir)
        
        temp_path = Path(temp_dir)
        main_results = list(temp_path.glob('**/all_regression_results*.json'))
        
        if main_results:
            with open(main_results[0], 'r') as f:
                data = json.load(f)
            
            all_datasets = set()
            dataset_task_mapping = {}
            
            for entry in data:
                dataset = entry.get('dataset_name', 'unknown')
                task_id = entry.get('task_id', 'unknown')
                if dataset != 'unknown':
                    all_datasets.add(dataset)
                    dataset_task_mapping[dataset] = task_id
            
            return all_datasets, dataset_task_mapping
    
    return set(), {}

def get_tabpfn_successful_datasets():
    """Get datasets where TabPFN v2 ran successfully."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open('results/tabular_baselines_reg.tar', 'r') as tar:
            tar.extractall(temp_dir)
        
        temp_path = Path(temp_dir)
        eval_files = list(temp_path.glob('**/all_evaluation_results*.json'))
        
        successful_datasets = set()
        dataset_task_mapping = {}
        
        for eval_file in eval_files:
            with open(eval_file, 'r') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    for entry in data:
                        if entry.get('model_name') == 'tabpfn_v2':
                            dataset = entry.get('dataset_name', 'unknown')
                            task_id = entry.get('dataset_id', entry.get('task_id', 'unknown'))
                            if dataset != 'unknown':
                                successful_datasets.add(dataset)
                                dataset_task_mapping[dataset] = task_id
        
        return successful_datasets, dataset_task_mapping

def main():
    """Main function."""
    print("🔍 Finding datasets where TabPFN v2 failed to run...")
    
    # Get all datasets from CLAM (complete set)
    all_datasets, clam_task_mapping = get_all_datasets_from_clam()
    print(f"Total datasets found (from CLAM): {len(all_datasets)}")
    
    # Get successful TabPFN datasets
    tabpfn_successful, tabpfn_task_mapping = get_tabpfn_successful_datasets()
    print(f"TabPFN v2 successful datasets: {len(tabpfn_successful)}")
    
    # Find failed datasets
    failed_datasets = all_datasets - tabpfn_successful
    print(f"TabPFN v2 failed datasets: {len(failed_datasets)}")
    
    if failed_datasets:
        print(f"\n❌ Datasets where TabPFN v2 FAILED to run ({len(failed_datasets)} datasets):")
        print("="*70)
        
        failed_list = sorted(list(failed_datasets))
        for i, dataset in enumerate(failed_list, 1):
            task_id = clam_task_mapping.get(dataset, 'unknown')
            print(f"{i:2d}. {dataset:45s} (task_id: {task_id})")
    
    if tabpfn_successful:
        print(f"\n✅ Datasets where TabPFN v2 SUCCEEDED ({len(tabpfn_successful)} datasets):")
        print("="*70)
        
        successful_list = sorted(list(tabpfn_successful))
        for i, dataset in enumerate(successful_list, 1):
            task_id = tabpfn_task_mapping.get(dataset, 'unknown')
            print(f"{i:2d}. {dataset:45s} (task_id: {task_id})")
    
    # Summary statistics
    print(f"\n📊 SUMMARY:")
    print(f"Total datasets: {len(all_datasets)}")
    print(f"TabPFN v2 successes: {len(tabpfn_successful)} ({len(tabpfn_successful)/len(all_datasets)*100:.1f}%)")
    print(f"TabPFN v2 failures: {len(failed_datasets)} ({len(failed_datasets)/len(all_datasets)*100:.1f}%)")

if __name__ == "__main__":
    main()