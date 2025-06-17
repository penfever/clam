#!/usr/bin/env python3
"""
Test to verify that embeddings are generated deterministically across multiple runs.
This test runs the evaluate_on_dataset.py script twice with the same parameters
and verifies that the generated embeddings are identical.
"""

import os
import sys
import subprocess
import numpy as np
import tempfile
import shutil
import argparse
from pathlib import Path

def run_evaluation(output_dir, dataset_id="3", seed=42):
    """Run evaluate_on_dataset.py and return the path to generated embeddings."""
    
    # Get the path to the examples directory
    script_dir = Path(__file__).parent.parent
    evaluate_script = script_dir / "examples" / "evaluate_on_dataset.py"
    
    # Create command
    cmd = [
        sys.executable,
        str(evaluate_script),
        "--model_id", "random_forest",  # Use RandomForest for speed
        "--dataset_ids", str(dataset_id),
        "--seed", str(seed),
        "--embedding_cache_dir", output_dir,
        "--force_recompute_embeddings",  # Force recomputation to test determinism
        "--max_test_samples", "300",  # Limit samples for faster testing
        "--num_few_shot_examples", "50",  # Smaller prefix for faster testing
        "--embedding_size", "100"  # Smaller embeddings for faster testing
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running evaluation:")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Evaluation failed with return code {result.returncode}")
    
    # Find the generated embedding file
    embedding_files = list(Path(output_dir).glob(f"{dataset_id}_tabpfn_embeddings_*.npz"))
    if not embedding_files:
        raise RuntimeError(f"No embedding files found in {output_dir}")
    
    return embedding_files[0]

def compare_embeddings(file1, file2):
    """Compare two embedding files and return whether they are identical."""
    
    # Load both files
    data1 = np.load(file1, allow_pickle=True)
    data2 = np.load(file2, allow_pickle=True)
    
    # Check if keys match
    keys1 = sorted(data1.keys())
    keys2 = sorted(data2.keys())
    
    if keys1 != keys2:
        print(f"Keys mismatch: {keys1} vs {keys2}")
        return False
    
    all_identical = True
    differences = {}
    
    for key in keys1:
        arr1 = data1[key]
        arr2 = data2[key]
        
        # Handle metadata dictionary specially
        if key == "metadata":
            # Extract the dictionaries
            meta1 = arr1.item() if hasattr(arr1, 'item') else arr1
            meta2 = arr2.item() if hasattr(arr2, 'item') else arr2
            
            # Compare non-timestamp fields
            for field in meta1:
                if field in ["timestamp", "date"]:
                    continue  # Skip timestamp fields
                if field not in meta2 or meta1[field] != meta2[field]:
                    print(f"Metadata field '{field}' differs: {meta1.get(field)} vs {meta2.get(field)}")
                    differences[f"metadata.{field}"] = (meta1.get(field), meta2.get(field))
                    all_identical = False
        else:
            # Compare arrays
            if not np.array_equal(arr1, arr2):
                all_identical = False
                if arr1.shape == arr2.shape:
                    max_diff = np.max(np.abs(arr1 - arr2))
                    mean_diff = np.mean(np.abs(arr1 - arr2))
                    differences[key] = {
                        "shape": arr1.shape,
                        "max_diff": max_diff,
                        "mean_diff": mean_diff
                    }
                else:
                    differences[key] = {
                        "shape1": arr1.shape,
                        "shape2": arr2.shape
                    }
    
    return all_identical, differences

def main():
    parser = argparse.ArgumentParser(description="Test embedding generation determinism")
    parser.add_argument("--dataset_id", type=str, default="3", help="OpenML dataset ID to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Testing Embedding Generation Determinism")
    print("=" * 70)
    
    # Create temporary directories for outputs
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        print(f"\nRunning first evaluation...")
        embedding_file1 = run_evaluation(tmpdir1, args.dataset_id, args.seed)
        print(f"First embedding file: {embedding_file1}")
        
        print(f"\nRunning second evaluation...")
        embedding_file2 = run_evaluation(tmpdir2, args.dataset_id, args.seed)
        print(f"Second embedding file: {embedding_file2}")
        
        print(f"\nComparing embeddings...")
        identical, differences = compare_embeddings(embedding_file1, embedding_file2)
        
        if identical:
            print("\n✅ SUCCESS: Embeddings are identical across runs!")
            print("The determinism fix is working correctly.")
        else:
            print("\n❌ FAILURE: Embeddings differ across runs!")
            print("\nDifferences found:")
            for key, diff in differences.items():
                print(f"\n  {key}:")
                if isinstance(diff, dict):
                    for k, v in diff.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"    {diff}")
            
            # Copy the differing files for debugging
            debug_dir = Path(__file__).parent / "debug_embeddings"
            debug_dir.mkdir(exist_ok=True)
            shutil.copy2(embedding_file1, debug_dir / "run1_embeddings.npz")
            shutil.copy2(embedding_file2, debug_dir / "run2_embeddings.npz")
            print(f"\nDebug files saved to: {debug_dir}")
            
            sys.exit(1)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()