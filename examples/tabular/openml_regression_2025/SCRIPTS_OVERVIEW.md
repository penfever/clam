# Scripts Overview: OpenML Regression 2025 Suite

This document provides detailed information about each script in the OpenML Regression 2025 evaluation suite.

## Script Hierarchy and Dependencies

```
Core Data Flow:
OpenML API → Task Fetching → Model Training/Evaluation → Results Analysis

Script Dependencies:
1. run_openml_regression_2025_tabular.py (standalone)
2. run_openml_regression_2025_llm_baselines_tabular.py (standalone)
3. analyze_regression_2025_results_wandb_tabular.py (depends on W&B data from 1,2)
```

## Detailed Script Descriptions

### 1. run_openml_regression_2025_tabular.py

**Purpose**: Main orchestration script for CLAM model training and evaluation on regression tasks.

**Key Features**:
- Fetches tasks from New OpenML Suite 2025 regression collection (study_id=455)
- Trains CLAM models with regression-specific adaptations
- Handles multiple train/test splits for robust evaluation
- Logs comprehensive experiment metadata

**Usage Patterns**:
```bash
# Full suite evaluation
python run_openml_regression_2025_tabular.py --clam_repo_path /path/to/clam --output_dir ./results

# Subset evaluation with custom parameters
python run_openml_regression_2025_tabular.py \
    --clam_repo_path /path/to/clam \
    --output_dir ./results \
    --start_idx 10 --end_idx 20 \
    --num_splits 5 \
    --total_steps 3000

# Skip training, evaluation only
python run_openml_regression_2025_tabular.py \
    --clam_repo_path /path/to/clam \
    --output_dir ./results \
    --skip_training
```

**Key Arguments**:
- `--clam_repo_path`: Path to CLAM repository (required)
- `--num_splits`: Number of train/test splits (default: 3)
- `--skip_training/--skip_evaluation`: Skip specific phases
- `--start_idx/--end_idx`: Process subset of tasks
- `--task_ids`: Comma-separated list of specific task IDs

**Output Structure**:
```
output_dir/
├── task_12345/
│   ├── split_0/
│   │   ├── model/           # Trained CLAM model
│   │   └── evaluation/      # Evaluation results
│   ├── split_1/
│   └── task_info_*.json     # Task metadata
```

**Regression Adaptations**:
- MSE-based early stopping (threshold: 0.1)
- Regression-specific task type specification
- Continuous target handling
- Adapted training parameters for regression objectives

---

### 2. run_openml_regression_2025_llm_baselines_tabular.py

**Purpose**: Comprehensive evaluation of LLM baselines on regression tasks with advanced visualization options.

**Supported Models**:
- **CLAM-t-SNE**: Visualization-driven regression with VLM reasoning
- **TabLLM**: LLM-based tabular regression
- **JOLT**: Joint learning of tabular representations
- **Tabula-8B**: Large-scale tabular model (experimental)

**Advanced Features**:
- 3D t-SNE/UMAP visualizations
- KNN connection analysis
- Detailed output logging with visualization cadence
- Regression-specific VLM prompting

**Usage Patterns**:
```bash
# All baselines with default settings
python run_openml_regression_2025_llm_baselines_tabular.py \
    --clam_repo_path /path/to/clam \
    --output_dir ./llm_results

# CLAM-t-SNE with advanced visualization
python run_openml_regression_2025_llm_baselines_tabular.py \
    --clam_repo_path /path/to/clam \
    --output_dir ./llm_results \
    --models clam_tsne \
    --use_3d \
    --use_knn_connections \
    --nn_k 10 \
    --save_detailed_outputs

# TabLLM with specific model configuration
python run_openml_regression_2025_llm_baselines_tabular.py \
    --clam_repo_path /path/to/clam \
    --output_dir ./llm_results \
    --models tabllm \
    --model_id "Qwen/Qwen2.5-7B-Instruct" \
    --batch_size 2
```

**Key Arguments**:
- `--models`: Space-separated list of models to evaluate
- `--use_3d`: Enable 3D visualizations
- `--use_knn_connections`: Show nearest neighbor connections
- `--nn_k`: Number of nearest neighbors for analysis
- `--save_detailed_outputs`: Save comprehensive visualization outputs
- `--max_vlm_image_size`: Resolution for VLM input images
- `--force_rerun`: Overwrite existing results

**Model-Specific Configuration**:

**CLAM-t-SNE**:
- Visualization parameters: `--use_3d`, `--nn_k`, `--use_knn_connections`
- Image settings: `--max_vlm_image_size`, `--image_dpi`
- Output control: `--save_outputs`, `--visualization_save_cadence`

**TabLLM**:
- Model selection: `--model_id`
- Generation: `--max_tokens`, `--batch_size`
- Prompting: Automatic regression prompt adaptation

**JOLT**:
- Standard configuration with regression loss adaptation

**Output Structure**:
```
output_dir/
├── task_12345/
│   ├── split_0/
│   │   ├── clam_tsne_results/
│   │   │   ├── aggregated_results.json
│   │   │   ├── visualizations/
│   │   │   └── llm_outputs/
│   │   ├── tabllm_results/
│   │   └── jolt_results/
│   └── task_info_*.json
├── regression_tasks_metadata_report.json
```

---

### 3. analyze_regression_2025_results_wandb_tabular.py

**Purpose**: Comprehensive analysis and visualization of regression experiment results from Weights & Biases.

**Analysis Features**:
- Statistical significance testing (Wilcoxon signed-rank)
- Performance correlation analysis between models
- Comprehensive visualization suite
- Effect size calculations and confidence intervals

**Key Visualizations**:
1. **Model Performance Comparison**: Boxplots across all regression metrics
2. **Performance Heatmap**: Model performance across different tasks
3. **Correlation Matrix**: Inter-model performance correlations
4. **Summary Statistics**: Mean performance with error bars

**Usage Patterns**:
```bash
# Basic analysis with default settings
python analyze_regression_2025_results_wandb_tabular.py \
    --project_name clam-regression-llm-baselines-2025 \
    --output_dir ./analysis

# Comprehensive analysis with raw data export
python analyze_regression_2025_results_wandb_tabular.py \
    --project_name clam-regression-llm-baselines-2025 \
    --output_dir ./analysis \
    --save_raw_data \
    --min_runs_per_model 10 \
    --confidence_level 0.99
```

**Key Arguments**:
- `--project_name`: W&B project name pattern to search
- `--entity`: W&B entity/team name
- `--min_runs_per_model`: Minimum runs required for model inclusion
- `--confidence_level`: Statistical confidence level (default: 0.95)
- `--save_raw_data`: Export raw extracted data as CSV

**Regression Metrics Analyzed**:
- **MSE (Mean Squared Error)**: Primary regression metric
- **RMSE (Root Mean Squared Error)**: Interpretable error scale
- **MAE (Mean Absolute Error)**: Robust error measure
- **R² (R-squared)**: Proportion of variance explained

**Statistical Analysis**:
- **Pairwise Comparisons**: Wilcoxon signed-rank tests between models
- **Effect Sizes**: Cohen's d calculations for practical significance
- **Confidence Intervals**: Bootstrap-based performance estimates
- **Correlation Analysis**: Inter-model performance relationships

**Output Files**:
```
analysis_dir/
├── performance_statistics.csv          # Detailed performance stats
├── statistical_tests.json             # Significance test results
├── analysis_summary.json              # High-level summary
├── raw_experiment_data.csv            # Raw W&B data (optional)
├── model_performance_comparison.png    # Boxplot comparisons
├── performance_heatmap.png            # Task vs Model heatmap
├── model_correlation.png              # Correlation matrix
└── performance_summary.png            # Summary statistics
```

---

## Cross-Script Integration

### Workflow Integration

1. **Data Generation Phase**:
   ```bash
   # Generate CLAM training data
   python run_openml_regression_2025_tabular.py --clam_repo_path /path/to/clam --output_dir ./clam_results
   
   # Generate LLM baseline data
   python run_openml_regression_2025_llm_baselines_tabular.py --clam_repo_path /path/to/clam --output_dir ./llm_results
   ```

2. **Analysis Phase**:
   ```bash
   # Analyze all results
   python analyze_regression_2025_results_wandb_tabular.py --project_name clam-regression-* --output_dir ./analysis
   ```

### Shared Configuration Patterns

**Common Arguments Across Scripts**:
- `--seed`: Reproducible random seed
- `--num_splits`: Number of train/test splits
- `--task_ids`: Subset of tasks to process
- `--output_dir`: Results directory
- `--wandb_project`: W&B project naming

**Consistent Naming Conventions**:
- Task directories: `task_{task_id}/`
- Split directories: `split_{split_idx}/`
- Timestamp format: `YYYY-MM-DD_HH-MM-SS`
- Metric naming: `mse`, `rmse`, `mae`, `r2`

### Result Compatibility

**JSON Schema Consistency**:
All scripts maintain compatible JSON schemas for:
- Task metadata
- Model performance metrics
- Experimental configuration
- Statistical analysis results

**W&B Integration**:
- Consistent project naming patterns
- Standardized metric logging
- Compatible run metadata
- Cross-script result aggregation

---

## Performance Considerations

### Computational Requirements

**Script 1 (Training)**:
- GPU: Recommended for CLAM training
- RAM: 8-16GB depending on dataset size
- Time: 1-4 hours per task (varies by dataset size)

**Script 2 (LLM Baselines)**:
- GPU: Required for efficient VLM inference
- RAM: 16-32GB for larger models
- Time: 30 minutes - 2 hours per task/model combination

**Script 3 (Analysis)**:
- CPU: Sufficient for analysis computations
- RAM: 4-8GB for large result sets
- Time: 5-15 minutes for comprehensive analysis

### Optimization Strategies

1. **Parallel Processing**:
   - Run different tasks in parallel using `--start_idx`/`--end_idx`
   - Distribute model evaluations across multiple GPUs
   - Use W&B parallel experiment tracking

2. **Resource Management**:
   - Monitor GPU memory usage with larger models
   - Adjust batch sizes based on available memory
   - Use appropriate image resolutions for VLM inputs

3. **Caching Strategies**:
   - CLAM automatically caches embeddings
   - Skip recomputation with `--force_rerun` control
   - Reuse W&B data for multiple analysis runs

---

## Debugging and Troubleshooting

### Common Issues and Solutions

**OpenML API Errors**:
```bash
# Check OpenML connectivity
python -c "import openml; print(openml.config.server)"

# Verify study exists
python -c "import openml; print(openml.study.get_suite(455).tasks)"
```

**W&B Authentication**:
```bash
# Login to W&B
wandb login

# Verify project access
wandb project list
```

**Memory Issues**:
- Reduce `--batch_size` for training/inference
- Lower `--max_vlm_image_size` for visualization
- Use `--gradient_accumulation_steps` for training

**Model Loading Errors**:
- Verify CLAM installation: `pip install -e ".[vision,audio,api]"`
- Check model availability: Test with smaller models first
- Ensure CUDA compatibility for GPU models

### Logging and Monitoring

**Log File Locations**:
- Script 1: `openml_regression_2025_run.log`
- Script 2: `openml_regression_2025_llm_baselines.log`
- Script 3: Standard output with structured logging

**Monitoring Commands**:
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Track disk usage
du -sh output_dir/

# Monitor W&B uploads
wandb status
```