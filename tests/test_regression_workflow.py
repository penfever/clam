#!/usr/bin/env python
"""
Test script to verify the end-to-end regression workflow.
Tests TabLLM and JOLT on the Boston housing dataset.
"""

import os
import sys
import tempfile
import logging

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_regression_workflow():
    """Test the regression workflow with a simple OpenML regression dataset."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing regression workflow in {temp_dir}")
        
        # Test command for TabLLM and JOLT on Boston housing
        cmd = [
            "python", "examples/tabular/evaluate_llm_baselines_tabular.py",
            "--dataset_ids", "531",  # Boston housing dataset
            "--models", "tabllm", "jolt",  # Test both TabLLM and JOLT
            "--output_dir", temp_dir,
            "--max_test_samples", "50",  # Keep it small for testing
            "--num_few_shot_examples", "8",  # Small number for quick test
            "--timeout_minutes", "10",  # Short timeout
            "--device", "cpu",  # Use CPU to avoid GPU issues in testing
            "--skip_classification",  # Only test regression
            "--skip_missing_metadata"  # Skip metadata validation for testing
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=current_dir)
            
            print("\n" + "="*50)
            print("STDOUT:")
            print("="*50)
            print(result.stdout)
            
            print("\n" + "="*50)
            print("STDERR:")
            print("="*50)
            print(result.stderr)
            
            print("\n" + "="*50)
            print(f"Return code: {result.returncode}")
            print("="*50)
            
            # Check if results files were created
            import glob
            result_files = glob.glob(os.path.join(temp_dir, "*.json"))
            print(f"\nResult files created: {result_files}")
            
            # Look for regression-specific metrics in results
            if result_files:
                import json
                for result_file in result_files:
                    try:
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                        
                        print(f"\nAnalyzing {result_file}:")
                        
                        if isinstance(results, list):
                            for result in results:
                                if 'task_type' in result:
                                    print(f"  Model: {result.get('model_name', 'unknown')}")
                                    print(f"  Task type: {result.get('task_type', 'unknown')}")
                                    print(f"  R² score: {result.get('r2_score', 'N/A')}")
                                    print(f"  MAE: {result.get('mae', 'N/A')}")
                                    print(f"  RMSE: {result.get('rmse', 'N/A')}")
                                    print()
                        
                    except Exception as e:
                        print(f"Error reading {result_file}: {e}")
            
            if result.returncode == 0:
                print("✅ Regression workflow test PASSED!")
                return True
            else:
                print("❌ Regression workflow test FAILED!")
                return False
                
        except Exception as e:
            print(f"Error running test: {e}")
            return False

if __name__ == "__main__":
    success = test_regression_workflow()
    sys.exit(0 if success else 1)