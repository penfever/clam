# OpenML Evaluation Framework

This directory contains a shared framework for evaluating LLM baselines on OpenML collections. The framework eliminates code duplication and ensures consistency across different OpenML evaluation scripts.

## Framework Components

### Core Modules

1. **`openml_orchestration.py`** - Main orchestration logic
   - `OpenMLEvaluationOrchestrator` - Handles command construction, task processing, and results aggregation
   - `handle_metadata_validation()` - Shared metadata validation and filtering
   - `generate_metadata_report()` - Creates metadata coverage reports

2. **`openml_args.py`** - Shared argument parsing
   - `parse_openml_collection_args()` - Configurable argument parser for collections
   - `get_cc18_args()` - Pre-configured CC18 arguments
   - `get_regression_2025_args()` - Pre-configured regression 2025 arguments

3. **`openml_collections.py`** - Task collection fetchers
   - `get_openml_cc18_tasks()` - Fetch CC18 tasks
   - `get_openml_regression_2025_tasks()` - Fetch regression 2025 tasks
   - `get_openml_collection_tasks()` - Generic collection fetcher

4. **`openml_template.py`** - Template for new collections

### Example Scripts

- **`openml_cc18/run_openml_cc18_llm_baselines_tabular_refactored.py`** - CC18 evaluation using shared framework
- **`openml_regression_2025/run_openml_regression_2025_llm_baselines_tabular_refactored.py`** - Regression 2025 evaluation using shared framework

## Creating a New OpenML Collection Script

1. **Copy the template:**
   ```bash
   cp examples/tabular/openml_template.py examples/tabular/your_collection/run_your_collection_llm_baselines.py
   ```

2. **Customize the settings:**
   ```python
   COLLECTION_NAME = "your_collection_name"
   STUDY_ID = 123  # Your OpenML study ID
   TASK_TYPE = "classification"  # or "regression"
   DEFAULT_MODELS = ["tabllm", "jolt", "clam_tsne"]
   ```

3. **Add collection-specific logic (if needed):**
   - Custom task fetching in `get_collection_tasks()`
   - Special preprocessing or filtering
   - Collection-specific metadata handling

4. **Run your script:**
   ```bash
   python examples/tabular/your_collection/run_your_collection_llm_baselines.py \
       --clam_repo_path . \
       --output_dir ./results
   ```

## Key Features

### Consistent Command Construction
All scripts generate identical evaluation commands with:
- Task-type specific flags (`--preserve_regression` for regression)
- Complete argument passing (backend, model configs, VLM settings)
- Proper model-specific argument handling (`--jolt_model`, `--tabllm_model`, etc.)

### Automatic Results Aggregation
- Task-type specific metrics (accuracy for classification, R²/MAE/MSE for regression)
- Cross-split and cross-task summaries
- JSON output with timestamps

### Metadata Validation
- Pre-flight metadata validation for all models
- Task filtering based on metadata availability
- Detailed coverage reports

### Error Handling
- Graceful handling of failed tasks
- Comprehensive logging
- Recovery from partial failures

## Usage Examples

### Basic Usage
```bash
# CC18 classification
python examples/tabular/openml_cc18/run_openml_cc18_llm_baselines_tabular_refactored.py \
    --clam_repo_path . \
    --output_dir ./cc18_results

# Regression 2025  
python examples/tabular/openml_regression_2025/run_openml_regression_2025_llm_baselines_tabular_refactored.py \
    --clam_repo_path . \
    --output_dir ./regression_results
```

### Advanced Usage
```bash
# Single model with custom settings
python examples/tabular/openml_regression_2025/run_openml_regression_2025_llm_baselines_tabular_refactored.py \
    --clam_repo_path . \
    --output_dir ./jolt_results \
    --models jolt \
    --jolt_model Qwen/Qwen2.5-3B-Instruct \
    --max_test_samples 200 \
    --backend transformers

# Metadata validation only
python examples/tabular/openml_cc18/run_openml_cc18_llm_baselines_tabular_refactored.py \
    --clam_repo_path . \
    --output_dir ./validation \
    --validate_metadata_only

# Subset of tasks
python examples/tabular/openml_regression_2025/run_openml_regression_2025_llm_baselines_tabular_refactored.py \
    --clam_repo_path . \
    --output_dir ./subset_results \
    --start_idx 10 \
    --end_idx 20
```

## Migration Guide

### From Legacy Scripts

1. **Replace individual functions** with orchestrator calls:
   ```python
   # Old
   evaluate_baseline_on_task(task, split_idx, model_name, args)
   
   # New
   orchestrator = OpenMLEvaluationOrchestrator(args, task_type="regression")
   orchestrator.process_task(task)
   ```

2. **Use shared argument parsing:**
   ```python
   # Old
   parser = argparse.ArgumentParser(...)
   # ... lots of argument definitions
   
   # New
   args = get_regression_2025_args()  # or get_cc18_args()
   ```

3. **Use shared task fetching:**
   ```python
   # Old
   def get_my_collection_tasks(): ...
   
   # New
   tasks = get_openml_collection_tasks("my_collection", study_id=123)
   ```

### Benefits of Migration

- **Consistency**: All scripts use identical logic and argument handling
- **Maintainability**: Bug fixes and improvements apply to all scripts
- **Extensibility**: Easy to add new collections or modify behavior
- **Testing**: Shared code is easier to test and validate

## Configuration

### Task Types
- `"classification"` - Does not pass `--preserve_regression`
- `"regression"` - Automatically passes `--preserve_regression`

### Default Models
- **Classification**: `["tabllm", "tabula_8b", "jolt", "clam_tsne"]`
- **Regression**: `["clam_tsne", "tabllm", "jolt"]` (excludes tabula_8b)

### Argument Defaults
- `num_few_shot_examples`: 16
- `preserve_regression`: True (for regression collections)
- `wandb_project`: Auto-generated based on collection name
- `output_dir`: Auto-generated based on collection name

## Troubleshooting

### Common Issues

1. **Missing arguments**: Ensure your script passes through all required arguments
2. **Path issues**: Use absolute paths for `--clam_repo_path`
3. **Model loading**: Check that model names match expected arguments (`jolt_model` vs `model_id`)
4. **Task fetching**: Verify OpenML study IDs and collection names

### Debug Mode
Add verbose logging to debug issues:
```python
logging.getLogger().setLevel(logging.DEBUG)
```

### Validation
Test argument parsing and command construction:
```python
from examples.tabular.openml_orchestration import OpenMLEvaluationOrchestrator
orchestrator = OpenMLEvaluationOrchestrator(args, task_type="regression")
cmd, _ = orchestrator.build_evaluation_command(task, 0)
print(" ".join(cmd))
```