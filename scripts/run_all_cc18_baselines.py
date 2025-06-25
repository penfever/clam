#!/usr/bin/env python3
"""
Wrapper script to run baselines on all OpenML CC18 datasets.
This script properly handles the task ID requirements.
"""

import subprocess
import sys
import os
import argparse

# Complete list of OpenML CC18 task IDs (with duplicates removed)
CC18_TASK_IDS = [
    3573, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3917, 3918,
    3950, 3954, 7592, 7593, 9914, 9946, 9957, 9960, 9961, 9962, 9964, 9965, 9966, 9967, 9968,
    9970, 9971, 9976, 9977, 9978, 9979, 9980, 9981, 9983, 9985, 9986, 9987, 10060, 10061,
    10064, 10065, 10066, 10067, 10068, 10069, 10070, 10071, 10072, 10073, 10074, 10075, 10076,
    10077, 10078, 10079, 10080, 10081, 10082, 10083, 10084, 10085, 10086, 10087, 10088, 10089,
    10090, 10092, 10093, 10096, 10097, 10098, 10099, 10100, 10101, 14954, 14965, 14969, 14970,
    125920, 125921, 125922, 125923, 125928, 125929, 125930, 125931, 125932, 125933, 125934,
    34536, 34537, 34539, 146574
]

# Remove duplicates and sort
CC18_TASK_IDS = sorted(list(set(CC18_TASK_IDS)))

def parse_args():
    parser = argparse.ArgumentParser(description="Run baselines on all OpenML CC18 datasets")
    parser.add_argument("--clam_repo_path", type=str, required=True, help="Path to CLAM repository")
    parser.add_argument("--output_dir", type=str, default="./openml_cc18_baseline_results", help="Output directory")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index in task list")
    parser.add_argument("--end_idx", type=int, default=None, help="End index in task list")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--num_splits", type=int, default=3, help="Number of splits per task")
    parser.add_argument("--max_train_samples", type=int, help="Maximum training samples")
    parser.add_argument("--max_test_samples", type=int, help="Maximum test samples")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Apply start/end indices if specified
    task_ids = CC18_TASK_IDS[args.start_idx:args.end_idx]
    
    # Convert task IDs to comma-separated string
    task_ids_str = ",".join(map(str, task_ids))
    
    print(f"Running baselines on {len(task_ids)} OpenML CC18 tasks...")
    print(f"Task indices: {args.start_idx} to {args.end_idx if args.end_idx else len(CC18_TASK_IDS)}")
    print(f"Total CC18 tasks: {len(CC18_TASK_IDS)}")
    print(f"Task IDs: {task_ids_str[:100]}...")  # Show first 100 chars
    
    # Find the script path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clam_repo = args.clam_repo_path
    baseline_script = os.path.join(clam_repo, "examples/tabular/openml_cc18/run_openml_cc18_baselines_tabular.py")
    
    # Build command
    cmd = [
        sys.executable,
        baseline_script,
        "--clam_repo_path", args.clam_repo_path,
        "--task_ids", task_ids_str,
        "--output_dir", args.output_dir,
        "--num_splits", str(args.num_splits)
    ]
    
    # Add optional arguments
    if args.no_wandb:
        cmd.append("--no_wandb")
    if args.max_train_samples:
        cmd.extend(["--max_train_samples", str(args.max_train_samples)])
    if args.max_test_samples:
        cmd.extend(["--max_test_samples", str(args.max_test_samples)])
    
    print(f"\nExecuting command:")
    print(" ".join(cmd[:8]) + " ...")  # Show abbreviated command
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Successfully completed baselines for {len(task_ids)} tasks!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running command: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()