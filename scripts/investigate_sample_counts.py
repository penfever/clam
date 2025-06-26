#!/usr/bin/env python3
"""
Investigate sample counts in regression results to understand duplicates.
"""

import json
import tarfile
import tempfile
from pathlib import Path
from collections import defaultdict

def analyze_tar_file(tar_path: str, name: str):
    """Analyze a single tar file for duplicate results."""
    print(f"\n{'='*80}")
    print(f"Analyzing {name} from {tar_path}")
    print('='*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract tar file
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(temp_dir)
        
        # Find all JSON result files
        temp_path = Path(temp_dir)
        json_files = list(temp_path.glob('**/*.json'))
        
        print(f"Found {len(json_files)} JSON files")
        
        # Analyze main aggregated results file
        main_results_files = [f for f in json_files if 'all_regression_results' in f.name]
        if main_results_files:
            print(f"\nMain results file: {main_results_files[0].name}")
            with open(main_results_files[0], 'r') as f:
                data = json.load(f)
            
            # Count by dataset
            dataset_counts = defaultdict(int)
            dataset_task_counts = defaultdict(lambda: defaultdict(int))
            
            for entry in data:
                dataset = entry.get('dataset_name', 'unknown')
                task_id = entry.get('task_id', 'unknown')
                dataset_counts[dataset] += 1
                dataset_task_counts[dataset][str(task_id)] += 1
            
            print(f"Total entries in main file: {len(data)}")
            print(f"Unique datasets: {len(dataset_counts)}")
            
            # Show datasets with more than 3 entries
            duplicates = [(count, dataset) for dataset, count in dataset_counts.items() if count > 3]
            if duplicates:
                print("\nDatasets with >3 entries (expected 3 for 3 splits):")
                duplicates.sort(reverse=True)
                for count, dataset in duplicates[:10]:  # Show top 10
                    task_ids = list(dataset_task_counts[dataset].keys())
                    print(f"  {dataset}: {count} entries (task_ids: {', '.join(task_ids)})")
        
        # Count individual result files by type
        aggregated_files = [f for f in json_files if 'aggregated_results.json' in f.name]
        jolt_files = [f for f in json_files if 'jolt_results.json' in f.name]
        eval_files = [f for f in json_files if 'all_evaluation_results' in f.name]
        
        print(f"\nFile type counts:")
        print(f"  aggregated_results.json files: {len(aggregated_files)}")
        print(f"  jolt_results.json files: {len(jolt_files)}")
        print(f"  all_evaluation_results*.json files: {len(eval_files)}")
        
        # Analyze aggregated_results.json files
        if name == 'CLAM' and aggregated_files:
            print(f"\nAnalyzing {len(aggregated_files)} aggregated_results.json files:")
            total_from_aggregated = 0
            for f in aggregated_files[:5]:  # Sample first 5
                try:
                    with open(f, 'r') as file:
                        data = json.load(file)
                        if isinstance(data, dict):
                            # Count entries in the dict
                            count = len([k for k, v in data.items() if isinstance(v, dict) and 'r2_score' in v])
                            if count > 0:
                                print(f"  {f.parent.name}: {count} results")
                                total_from_aggregated += count
                except:
                    pass
            
            print(f"  (Total from sampled aggregated files: {total_from_aggregated})")

def main():
    """Main function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / "results"
    
    tar_files = {
        'CLAM': results_dir / "clam-reg.tar",
        'JOLT': results_dir / "jolt_reg.tar", 
        'Baselines': results_dir / "tabular_baselines_reg.tar"
    }
    
    for name, tar_path in tar_files.items():
        if tar_path.exists():
            analyze_tar_file(str(tar_path), name)

if __name__ == "__main__":
    main()