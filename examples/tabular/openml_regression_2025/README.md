# OpenML Regression 2025 Suite

This directory contains scripts for training and evaluating CLAM models and LLM baselines on the **New OpenML Suite 2025 regression collection** (study_id=455).

## Overview

The New OpenML Suite 2025 regression collection provides a comprehensive benchmark for evaluating machine learning models on regression tasks. This suite adapts the successful OpenML CC18 framework for continuous prediction problems.

## Key Features

- **Regression-focused evaluation**: Adapted metrics (MSE, RMSE, MAE, R²) instead of classification metrics
- **Multiple model support**: CLAM-t-SNE, TabLLM, JOLT, and traditional baselines
- **Visualization-driven approach**: t-SNE/UMAP visualizations for regression pattern analysis
- **Comprehensive analysis**: Statistical significance testing and performance correlation analysis
- **Reproducible experiments**: Standardized splits, seeds, and evaluation protocols

## Scripts Overview

### Core Evaluation Scripts

1. **`run_openml_regression_2025_tabular.py`**
   - Main script for training and evaluating CLAM models on regression tasks
   - Handles data loading, model training, and basic evaluation
   - Supports multiple splits and reproducible experiments

2. **`run_openml_regression_2025_llm_baselines_tabular.py`**
   - Evaluates LLM baselines (CLAM-t-SNE, TabLLM, JOLT) on regression tasks
   - Supports advanced visualization options (3D t-SNE, KNN connections)
   - Handles regression-specific VLM prompting

3. **`analyze_regression_2025_results_wandb_tabular.py`**
   - Comprehensive analysis of experimental results from Weights & Biases
   - Statistical significance testing between models
   - Performance visualization and correlation analysis

## Quick Start

### 1. Basic CLAM Training and Evaluation

```bash
# Train and evaluate CLAM on all regression tasks
python run_openml_regression_2025_tabular.py \
    --clam_repo_path /path/to/clam \
    --output_dir ./regression_results \
    --num_splits 3

# Train on specific tasks
python run_openml_regression_2025_tabular.py \
    --clam_repo_path /path/to/clam \
    --output_dir ./regression_results \
    --task_ids "12345,67890" \
    --num_splits 5
```

### 2. LLM Baseline Evaluation

```bash
# Evaluate all LLM baselines
python run_openml_regression_2025_llm_baselines_tabular.py \
    --clam_repo_path /path/to/clam \
    --output_dir ./llm_results \
    --models clam_tsne tabllm jolt

# CLAM-t-SNE with advanced visualization
python run_openml_regression_2025_llm_baselines_tabular.py \
    --clam_repo_path /path/to/clam \
    --output_dir ./llm_results \
    --models clam_tsne \
    --use_3d \
    --use_knn_connections \
    --nn_k 7 \
    --save_detailed_outputs
```

### 3. Results Analysis

```bash
# Analyze results from W&B
python analyze_regression_2025_results_wandb_tabular.py \
    --project_name clam-regression-llm-baselines-2025 \
    --output_dir ./analysis \
    --save_raw_data
```

## Configuration Options

### Model Selection
- `clam_tsne`: CLAM with t-SNE visualization for regression
- `tabllm`: TabLLM adapted for regression tasks
- `jolt`: JOLT baseline for regression
- `tabula_8b`: Tabula-8B (experimental regression support)

### Visualization Options (CLAM-t-SNE)
- `--use_3d`: Enable 3D t-SNE/UMAP visualizations
- `--use_knn_connections`: Show nearest neighbor connections
- `--nn_k`: Number of nearest neighbors (default: 7)
- `--max_vlm_image_size`: Image resolution for VLM input
- `--visualization_save_cadence`: Frequency of saving visualization outputs

### Regression-Specific Settings
- `--task_type regression`: Explicitly specify regression mode
- Adapted early stopping thresholds for MSE-based metrics
- Continuous target handling in data processing
- Regression-specific VLM prompting templates

## Output Structure

```
regression_results/
├── task_12345/                    # Per-task results
│   ├── split_0/
│   │   ├── model/                 # Trained CLAM model
│   │   ├── evaluation/            # Evaluation results
│   │   └── clam_tsne_results/     # CLAM-t-SNE specific outputs
│   ├── split_1/
│   └── task_info_TIMESTAMP.json   # Task metadata
├── analysis/                      # Analysis outputs
│   ├── performance_statistics.csv
│   ├── statistical_tests.json
│   ├── model_performance_comparison.png
│   ├── performance_heatmap.png
│   └── analysis_summary.json
└── regression_tasks_metadata_report.json
```

## Evaluation Metrics

### Primary Regression Metrics
- **MSE (Mean Squared Error)**: Primary optimization target
- **RMSE (Root Mean Squared Error)**: Interpretable error magnitude
- **MAE (Mean Absolute Error)**: Robust to outliers
- **R² (Coefficient of Determination)**: Proportion of variance explained

### Statistical Analysis
- Wilcoxon signed-rank tests for paired comparisons
- Effect size calculations (Cohen's d)
- Confidence intervals for performance estimates
- Correlation analysis between model performances

## Differences from Classification (CC18)

### Adapted Components
1. **Metrics**: Regression metrics instead of accuracy/F1
2. **VLM Prompting**: Continuous value prediction prompts
3. **Visualization**: Regression-appropriate t-SNE interpretations
4. **Target Processing**: Continuous variable handling
5. **Early Stopping**: MSE-based thresholds

### Maintained Features
- Multiple train/test splits for robust evaluation
- Standardized experimental protocols
- Comprehensive logging and result tracking
- Statistical significance testing framework

## Requirements

### Core Dependencies
```bash
# CLAM installation
pip install -e ".[vision,audio,api]"

# Additional requirements
pip install openml wandb
pip install matplotlib seaborn scipy pandas
pip install scikit-learn numpy torch

# Optional: Tabula-8B support
pip install git+https://github.com/penfever/rtfm.git
```

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended for larger models)
- 16GB+ RAM for larger datasets
- W&B account for experiment tracking

## Best Practices

### Experiment Configuration
1. **Reproducibility**: Always set `--seed` for consistent results
2. **Multiple Splits**: Use `--num_splits 3` or higher for robust evaluation
3. **Resource Management**: Monitor GPU memory with larger models
4. **Checkpointing**: Use `--save_best_model` for model persistence

### Performance Optimization
1. **Batch Sizing**: Adjust based on available GPU memory
2. **Early Stopping**: Use appropriate thresholds for regression (e.g., 0.1 for MSE)
3. **Feature Selection**: Apply `--feature_selection_threshold` for high-dimensional data
4. **Visualization Quality**: Balance `--image_dpi` and `--max_vlm_image_size` for VLM performance

### Analysis Guidelines
1. **Statistical Significance**: Use multiple runs and appropriate statistical tests
2. **Effect Sizes**: Consider practical significance alongside statistical significance
3. **Cross-Dataset Generalization**: Analyze performance patterns across different regression domains
4. **Visualization Interpretation**: Understand how regression patterns manifest in t-SNE space

## Troubleshooting

### Common Issues

1. **OpenML API Errors**: Check internet connection and OpenML server status
2. **W&B Authentication**: Run `wandb login` before experiments
3. **Memory Issues**: Reduce batch size or image resolution
4. **Missing Tasks**: Verify study_id=455 exists and contains regression tasks

### Performance Debugging

1. **Slow VLM Inference**: Check GPU utilization and model size
2. **Poor Regression Performance**: Verify target preprocessing and scaling
3. **Visualization Quality**: Adjust t-SNE parameters and image settings
4. **Statistical Power**: Ensure sufficient runs for meaningful comparisons

## Contributing

When adding new features or models:

1. Maintain regression-specific adaptations
2. Update evaluation metrics appropriately
3. Test with multiple regression datasets
4. Document parameter sensitivity
5. Verify statistical analysis compatibility

## Citation

If you use this regression benchmark suite, please cite:

```bibtex
@misc{openml_regression_2025,
    title={New OpenML Suite 2025 Regression Benchmark},
    author={OpenML Community},
    year={2025},
    url={https://new.openml.org/search?type=study&id=455}
}

@misc{clam_regression,
    title={CLAM: Classification Learning with Adaptive Multimodal Visualization for Regression Tasks},
    year={2025},
    note={Adapted for regression evaluation}
}
```