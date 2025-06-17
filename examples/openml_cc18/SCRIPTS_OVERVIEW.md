# OpenML CC18 Evaluation Scripts

This directory contains scripts for running extensive experiments on the OpenML CC18 collection of classification tasks. The scripts train CLAM models and evaluate them against standard baselines, generating comprehensive performance metrics.

## Overview of Files

- **`run_openml_cc18.py`**: Main Python script that automates the process of training and evaluating CLAM models on OpenML CC18 datasets.
  
- **`run_cc18_batch.sh`**: Helper shell script to run the Python script with various configurations, making it easy to run subsets of tasks or specific task IDs.
  
- **`analyze_cc18_results.py`**: Analysis script that processes experiment results, computes summary statistics, creates visualizations, and exports data for further analysis.
  
- **`OPENML_CC18_README.md`**: Detailed documentation of the experiment setup, usage instructions, and result organization.

## How It Works

1. The main script fetches all 72 tasks from the OpenML CC18 collection (study ID 99).
2. For each task, it trains a CLAM model on three different train/test splits.
3. It then evaluates each trained model and runs all standard baselines on the same splits.
4. Results are stored in a structured directory hierarchy, with each task and split having its own directory.
5. The analysis script compiles these results into summary statistics, comparative visualizations, and exportable data files.

## Usage

```bash
# Train and evaluate on all tasks
./run_cc18_batch.sh /path/to/clam/repo

# Train and evaluate on a subset of tasks (e.g., tasks 0-9)
./run_cc18_batch.sh /path/to/clam/repo 0 10

# Train and evaluate on specific task IDs
./run_cc18_batch.sh /path/to/clam/repo 0 0 "3573,3902,3903"

# Analyze results after experiments are complete
./analyze_cc18_results.py --results_dir ./openml_cc18_results
```

## Configuration

The scripts use the following default configurations:

**Training:**
- Batch size: 1
- Number of epochs: 5
- Final learning rate: 1e-5
- Mixup alpha: 0.0
- W&B entity: "nyu-dice-lab"

**Evaluation:**
- All baselines are run
- Only ground truth classes are considered
- Results are logged to W&B

## Output

The experiment generates a structured directory of results and the analysis produces:

1. Summary statistics for each model across all tasks
2. Comparative visualizations (boxplots, violin plots, heatmaps)
3. Win rate analyses for each performance metric
4. Exported CSV and JSON files for further analysis

## Requirements

- Python 3.7+
- OpenML (`pip install openml`)
- CLAM repository and its dependencies
- Weights & Biases account for logging results
- For analysis: pandas, matplotlib, seaborn