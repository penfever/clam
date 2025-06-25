#!/usr/bin/env python3
"""
Analyze regression results from all_regression_results_2025-06-25_05-13-20.json

This script reads the regression results JSON file and computes:
- Average R² score over all datasets
- Applies minimum R² score of 0 (ignores negative scores)
- Provides detailed breakdown by dataset

Usage:
    python scripts/analyze_regression_results.py
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import statistics

def analyze_regression_results(json_file_path: str):
    """
    Analyze regression results from JSON file.
    
    Args:
        json_file_path: Path to the regression results JSON file
    """
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"❌ Error: File not found: {json_file_path}")
        return
    
    print(f"📊 Analyzing regression results from: {json_file_path}")
    print("=" * 80)
    
    # Load the JSON data
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        return
    
    # Extract R² scores by dataset
    dataset_r2_scores = defaultdict(list)
    total_r2_scores = []
    negative_r2_count = 0
    total_entries = 0
    
    for entry in data:
        total_entries += 1
        
        # Extract dataset information
        dataset_name = entry.get('dataset_name', 'unknown')
        task_id = entry.get('task_id', 'unknown')
        dataset_id = entry.get('dataset_id', 'unknown')
        
        # Create dataset identifier
        dataset_key = f"{dataset_name} (task_id: {task_id})"
        
        # Extract R² score
        r2_score = entry.get('r2_score')
        
        if r2_score is not None:
            if r2_score < 0:
                negative_r2_count += 1
                print(f"⚠️  Negative R² score found for {dataset_key}: {r2_score:.4f} (will use 0.0)")
                r2_score = 0.0  # Apply minimum of 0
            
            dataset_r2_scores[dataset_key].append(r2_score)
            total_r2_scores.append(r2_score)
        else:
            print(f"⚠️  Missing R² score for {dataset_key}")
    
    # Print summary statistics
    print(f"\n📈 SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Total entries processed: {total_entries}")
    print(f"Entries with R² scores: {len(total_r2_scores)}")
    print(f"Negative R² scores found: {negative_r2_count}")
    print(f"Unique datasets: {len(dataset_r2_scores)}")
    
    if not total_r2_scores:
        print("❌ No R² scores found in the data!")
        return
    
    # Compute overall statistics
    avg_r2 = statistics.mean(total_r2_scores)
    median_r2 = statistics.median(total_r2_scores)
    min_r2 = min(total_r2_scores)
    max_r2 = max(total_r2_scores)
    std_r2 = statistics.stdev(total_r2_scores) if len(total_r2_scores) > 1 else 0
    
    print(f"\n🎯 OVERALL R² STATISTICS")
    print("=" * 50)
    print(f"Average R² score: {avg_r2:.6f}")
    print(f"Median R² score:  {median_r2:.6f}")
    print(f"Minimum R² score: {min_r2:.6f}")
    print(f"Maximum R² score: {max_r2:.6f}")
    print(f"Standard deviation: {std_r2:.6f}")
    
    # Dataset-by-dataset breakdown
    print(f"\n📋 DATASET BREAKDOWN")
    print("=" * 50)
    
    dataset_averages = []
    for dataset_key, scores in sorted(dataset_r2_scores.items()):
        dataset_avg = statistics.mean(scores)
        dataset_averages.append(dataset_avg)
        
        print(f"{dataset_key:40s}: {dataset_avg:.6f} ({len(scores)} samples)")
        
        # Show individual scores if multiple entries per dataset
        if len(scores) > 1:
            score_str = ", ".join([f"{s:.4f}" for s in scores])
            print(f"{'':42s}  Individual scores: [{score_str}]")
    
    # Compute average of dataset averages (in case datasets have different numbers of samples)
    if dataset_averages:
        avg_of_dataset_avgs = statistics.mean(dataset_averages)
        print(f"\n🎯 DATASET-WEIGHTED AVERAGE R² SCORE")
        print("=" * 50)
        print(f"Average of dataset averages: {avg_of_dataset_avgs:.6f}")
        print(f"(This treats each dataset equally regardless of sample count)")
    
    # Performance categories
    print(f"\n📊 PERFORMANCE CATEGORIES")
    print("=" * 50)
    
    excellent = sum(1 for r2 in total_r2_scores if r2 >= 0.9)
    good = sum(1 for r2 in total_r2_scores if 0.7 <= r2 < 0.9)
    fair = sum(1 for r2 in total_r2_scores if 0.5 <= r2 < 0.7)
    poor = sum(1 for r2 in total_r2_scores if 0.0 <= r2 < 0.5)
    
    total_scored = len(total_r2_scores)
    
    print(f"Excellent (R² ≥ 0.9): {excellent:4d} ({excellent/total_scored*100:.1f}%)")
    print(f"Good (0.7 ≤ R² < 0.9): {good:4d} ({good/total_scored*100:.1f}%)")
    print(f"Fair (0.5 ≤ R² < 0.7): {fair:4d} ({fair/total_scored*100:.1f}%)")
    print(f"Poor (0.0 ≤ R² < 0.5): {poor:4d} ({poor/total_scored*100:.1f}%)")
    
    print(f"\n✅ Analysis complete!")
    print(f"🎯 **FINAL ANSWER: Average R² score = {avg_r2:.6f}**")

def main():
    """Main function."""
    
    # Default path to the regression results file
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_file = project_root / "results" / "all_regression_results_2025-06-25_05-13-20.json"
    
    # Allow override via command line argument
    if len(sys.argv) > 1:
        results_file = Path(sys.argv[1])
    
    analyze_regression_results(str(results_file))

if __name__ == "__main__":
    main()