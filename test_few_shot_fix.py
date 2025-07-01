#!/usr/bin/env python3

import subprocess
import tempfile
import os
import json
import sys

# Add the clam directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_few_shot_orchestration():
    """Test that the orchestration script properly passes few-shot parameters."""
    
    print("=== Testing Few-Shot Parameter Passing ===")
    
    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Run the orchestration script with few-shot parameters
        cmd = [
            "python", "examples/tabular/openml_cc18/run_openml_cc18_baselines_tabular.py",
            "--clam_repo_path", current_dir,
            "--task_ids", "3",  # Just test on task 3 (kr-vs-kp)
            "--output_dir", temp_dir,
            "--num_few_shot_examples", "8",
            "--balanced_few_shot",
            "--no_wandb",  # Disable W&B for testing
            "--num_splits", "1"  # Just test one split
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            # Capture both stdout and stderr
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,  # 5 minute timeout
                cwd=current_dir
            )
            
            print(f"Return code: {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")
            
            if result.returncode != 0:
                print("❌ Command failed")
                return False
                
            # Check if output files were created
            task_dir = os.path.join(temp_dir, "task_3", "split_0", "baselines")
            if not os.path.exists(task_dir):
                print(f"❌ Output directory not created: {task_dir}")
                return False
                
            # Look for evaluation results
            results_files = [f for f in os.listdir(task_dir) if f.startswith("all_evaluation_results_")]
            if not results_files:
                print("❌ No evaluation results files found")
                return False
                
            # Read the results and check training sample size
            results_file = os.path.join(task_dir, results_files[0])
            with open(results_file, 'r') as f:
                results = json.load(f)
                
            if not results:
                print("❌ Empty results file")
                return False
                
            # Check the first model's training sample count
            first_result = results[0]
            num_train_samples = first_result['num_train_samples']
            
            print(f"✅ Found evaluation results")
            print(f"Training samples used: {num_train_samples}")
            
            # With balanced few-shot and 8 examples per class, and kr-vs-kp has 2 classes
            # we expect 8 * 2 = 16 training samples
            expected_samples = 16  # 8 examples per class × 2 classes
            
            if num_train_samples == expected_samples:
                print(f"✅ SUCCESS: Training samples ({num_train_samples}) matches expected few-shot size ({expected_samples})")
                return True
            else:
                print(f"❌ FAILURE: Training samples ({num_train_samples}) does not match expected few-shot size ({expected_samples})")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Command timed out")
            return False
        except Exception as e:
            print(f"❌ Error running command: {e}")
            return False

if __name__ == "__main__":
    success = test_few_shot_orchestration()
    if success:
        print("\n🎉 Test passed! Few-shot parameters are working correctly.")
    else:
        print("\n💥 Test failed! Few-shot parameters are not being respected.")
    
    sys.exit(0 if success else 1)