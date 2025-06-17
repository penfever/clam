# OpenML CC18 Experiments with LLATA

This directory contains scripts to run LLATA models on the OpenML CC18 collection of datasets. The OpenML CC18 collection consists of 72 classification tasks curated for benchmarking machine learning algorithms.

## Scripts

- `run_openml_cc18.py`: Main Python script that trains and evaluates LLATA models on OpenML CC18 datasets
- `run_cc18_batch.sh`: Helper bash script to run the Python script with different configurations

## Requirements

- Python 3.7+
- OpenML (`pip install openml`)
- LLATA repository and its dependencies
- Weights & Biases account

## Usage

### Running all tasks

```bash
./run_cc18_batch.sh /path/to/llata/repo
```

### Running a subset of tasks by index

```bash
./run_cc18_batch.sh /path/to/llata/repo 0 10  # Run first 10 tasks
```

### Running specific tasks by task ID

```bash
./run_cc18_batch.sh /path/to/llata/repo 0 0 "3573,3902,3903"  # Run tasks with specific IDs
```

### Advanced usage with Python script directly

For more control, you can use the Python script directly:

```bash
python run_openml_cc18.py --llata_repo_path /path/to/llata/repo --output_dir ./results --seed 42
```

Additional options:
- `--skip_training`: Only run evaluation on existing models
- `--skip_evaluation`: Only train models without evaluation
- `--model_id`: Specify a different model ID (default: "Qwen/Qwen2.5-3B-Instruct")
- `--num_splits`: Number of splits to use for each task (default: 3)
- `--wandb_project`: W&B project name (default: "llata-openml-cc18")

## Output Structure

Results are organized in the following directory structure:

```
openml_cc18_results/
├── task_3573/                  # Task ID
│   ├── task_info.json          # Task metadata
│   ├── split_0/                # First split
│   │   ├── model/              # Trained model
│   │   └── evaluation/         # Evaluation results
│   ├── split_1/                # Second split
│   │   ├── model/
│   │   └── evaluation/
│   └── split_2/                # Third split
│       ├── model/
│       └── evaluation/
├── task_3902/
│   └── ...
└── ...
```

## Training Configuration

The scripts use the following default configuration for training:
- Batch size: 1
- Number of epochs: 5
- Final learning rate: 1e-5
- Mixup alpha: 0.0
- W&B entity: "nyu-dice-lab"

## Evaluation Configuration

The scripts use the following default configuration for evaluation:
- All baselines are evaluated
- Only ground truth classes are considered
- Results are logged to W&B (entity: "nyu-dice-lab")

## Notes

- Each task-split combination gets a different random seed to ensure diversity
- Training and evaluation logs are saved to `openml_cc18_run.log`
- The process can be resumed if interrupted by using appropriate start/end indices