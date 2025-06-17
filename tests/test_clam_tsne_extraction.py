#!/usr/bin/env python
"""
Test script to verify that llata_tsne balanced_accuracy extraction and split detection work correctly.
"""

import pandas as pd
import sys
import os

# Add the llata module to path
sys.path.insert(0, "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/llata/llata")

from clam.utils.wandb_extractor import (
    fetch_wandb_data,
    extract_results_from_wandb,
    extract_model_metrics_from_summary,
    extract_variables_from_wandb_data
)

def test_llata_tsne_extraction():
    """Test llata_tsne metric extraction with both old and new formats."""
    
    print("Testing llata_tsne metric extraction...")
    
    # Test data with new hierarchical format (like what we found in the CSV)
    test_summary_new = {
        "llata_tsne/dataset/kr-vs-kp/accuracy": 0.85,
        "llata_tsne/dataset/kr-vs-kp/balanced_accuracy": 0.82,
        "llata_tsne/dataset/kr-vs-kp/f1_score": 0.83,
        "llata_tsne/dataset/kr-vs-kp/completed_samples": 200,
        "llata_tsne/dataset/balance-scale/accuracy": 0.75,
        "llata_tsne/dataset/balance-scale/balanced_accuracy": 0.72,
        "llata_tsne/dataset/balance-scale/f1_score": 0.73,
        "llata_tsne/dataset/balance-scale/completed_samples": 150,
    }
    
    # Test data with old format
    test_summary_old = {
        "llata_tsne_kr-vs-kp_accuracy": 0.85,
        "llata_tsne_kr-vs-kp_balanced_accuracy": 0.82,
        "llata_tsne_kr-vs-kp_f1_score": 0.83,
        "llata_tsne_kr-vs-kp_completed_samples": 200,
    }
    
    print("\n=== Testing NEW hierarchical format ===")
    metrics_new = extract_model_metrics_from_summary(test_summary_new, debug=True)
    print("Extracted metrics:", metrics_new)
    
    # Check if llata_tsne model was detected
    if "clam_tsne" in metrics_new:
        print("✓ llata_tsne model detected")
        llata_tsne_metrics = metrics_new["clam_tsne"]
        
        # Check for balanced_accuracy
        balanced_accuracy_found = False
        for key, value in llata_tsne_metrics.items():
            if "balanced_accuracy" in key:
                print(f"✓ Found balanced_accuracy: {key} = {value}")
                balanced_accuracy_found = True
        
        if not balanced_accuracy_found:
            print("✗ balanced_accuracy NOT found in llata_tsne metrics")
        
        # Check for multiple datasets (split detection proxy)
        datasets_found = set()
        for key in llata_tsne_metrics.keys():
            if "_" in key:
                dataset_name = key.split("_")[0]
                datasets_found.add(dataset_name)
        
        print(f"✓ Found {len(datasets_found)} datasets: {datasets_found}")
        
    else:
        print("✗ llata_tsne model NOT detected")
    
    print("\n=== Testing OLD format ===")
    metrics_old = extract_model_metrics_from_summary(test_summary_old, debug=True)
    print("Extracted metrics:", metrics_old)
    
    # Check if llata_tsne model was detected
    if "clam_tsne" in metrics_old:
        print("✓ llata_tsne model detected")
        llata_tsne_metrics = metrics_old["clam_tsne"]
        
        # Check for balanced_accuracy
        balanced_accuracy_found = False
        for key, value in llata_tsne_metrics.items():
            if "balanced_accuracy" in key:
                print(f"✓ Found balanced_accuracy: {key} = {value}")
                balanced_accuracy_found = True
        
        if not balanced_accuracy_found:
            print("✗ balanced_accuracy NOT found in llata_tsne metrics")
    else:
        print("✗ llata_tsne model NOT detected")

def test_variable_extraction():
    """Test variable extraction from wandb data."""
    
    print("\n=== Testing Variable Extraction ===")
    
    # Test new hierarchical format
    test_summary = {
        "llata_tsne/dataset/kr-vs-kp/accuracy": 0.85,
        "llata_tsne/dataset/kr-vs-kp/balanced_accuracy": 0.82,
    }
    test_config = {}
    test_run_name = "llata_tsne_task3_split1"
    
    variables = extract_variables_from_wandb_data(
        test_summary, test_config, test_run_name, "llm_baseline"
    )
    
    print("Extracted variables:", variables)
    
    if variables["dataset_name"] == "kr-vs-kp":
        print("✓ dataset_name extracted correctly")
    else:
        print(f"✗ dataset_name incorrect: expected 'kr-vs-kp', got '{variables['dataset_name']}'")
    
    if variables["task_id"] is not None:
        print(f"✓ task_id extracted/imputed: {variables['task_id']}")
    else:
        print("✗ task_id not extracted/imputed")

def test_real_wandb_data():
    """Test with real wandb data."""
    
    print("\n=== Testing with Real W&B Data ===")
    
    try:
        # Fetch a small amount of real data to test
        wandb_projects = ["llata-openml-cc18-baselines-alldata", "llata-fewshot-openml-cc18-hero2"]
        
        print("Fetching W&B data...")
        wandb_df = fetch_wandb_data("nyu-dice-lab", wandb_projects[:1])  # Just test with one project
        
        print(f"Fetched {len(wandb_df)} runs")
        
        # Look for llata_tsne runs
        llata_tsne_runs = []
        for _, row in wandb_df.iterrows():
            summary = row["summary"]
            for key in summary.keys():
                if "clam_tsne" in key:
                    llata_tsne_runs.append(row)
                    break
        
        print(f"Found {len(llata_tsne_runs)} llata_tsne runs")
        
        if llata_tsne_runs:
            # Test extraction on first llata_tsne run
            test_run = llata_tsne_runs[0]
            print(f"\nTesting run: {test_run['name']}")
            
            # Show some llata_tsne keys
            llata_tsne_keys = [k for k in test_run["summary"].keys() if "clam_tsne" in k]
            print(f"llata_tsne keys: {llata_tsne_keys[:5]}...")  # Show first 5
            
            # Test metric extraction
            metrics = extract_model_metrics_from_summary(test_run["summary"], debug=True)
            
            if "clam_tsne" in metrics:
                print("✓ llata_tsne metrics extracted successfully")
                llata_tsne_metrics = metrics["clam_tsne"]
                
                # Look for balanced_accuracy
                balanced_accuracy_keys = [k for k in llata_tsne_metrics.keys() if "balanced_accuracy" in k]
                if balanced_accuracy_keys:
                    print(f"✓ Found balanced_accuracy metrics: {balanced_accuracy_keys}")
                else:
                    print("✗ No balanced_accuracy metrics found")
                    
                print(f"Total llata_tsne metrics: {len(llata_tsne_metrics)}")
            else:
                print("✗ llata_tsne metrics NOT extracted")
        
    except Exception as e:
        print(f"Error testing real data: {e}")

if __name__ == "__main__":
    print("Testing llata_tsne extraction fixes...")
    
    test_llata_tsne_extraction()
    test_variable_extraction()
    test_real_wandb_data()
    
    print("\nTest completed!")