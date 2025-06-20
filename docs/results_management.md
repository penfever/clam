# CLAM Unified Results Management System

This document describes the new unified results management system for CLAM, which provides standardized, consistent storage and organization of experiment results across all modalities (tabular, vision, audio).

## Overview

The unified results management system addresses several issues with the previous approach:

- **Inconsistent directory structures** across different evaluation scripts
- **Mixed file naming conventions** and formats
- **Scattered metadata** and experimental parameters
- **Difficulty comparing results** across modalities and datasets
- **Manual result organization** and cleanup

The new system provides:

- **Standardized directory structure**: `<base>/results/<modality>/<dataset_id>/<model_name>/`
- **Comprehensive metadata tracking** with experimental parameters
- **Unified file formats** with backward compatibility
- **Automatic artifact management** for visualizations and outputs
- **Migration tools** for existing results
- **CLI utilities** for management and reporting

## Quick Start

### Basic Usage

```python
from clam.utils import get_results_manager, EvaluationResults, ExperimentMetadata

# Get the results manager
results_manager = get_results_manager()

# Create evaluation results
results = EvaluationResults(
    accuracy=0.8542,
    precision_macro=0.8234,
    recall_macro=0.8456,
    f1_macro=0.8344,
    status="completed"
)

# Create experiment metadata
metadata = ExperimentMetadata(
    model_name="tabllm",
    dataset_id="adult",
    modality="tabular",
    num_samples_train=1000,
    num_samples_test=200,
    k_shot=5,
    random_seed=42
)

# Save results
experiment_dir = results_manager.save_evaluation_results(
    model_name="tabllm",
    dataset_id="adult",
    modality="tabular",
    results=results,
    experiment_metadata=metadata
)
```

### Loading Results

```python
# Load specific experiment results
loaded_results = results_manager.load_evaluation_results(
    model_name="tabllm",
    dataset_id="adult", 
    modality="tabular"
)

# List all experiments
experiments = results_manager.list_experiments()

# Filter by modality
tabular_experiments = results_manager.list_experiments(modality="tabular")
```

### Backward Compatibility

The system maintains backward compatibility with existing scripts:

```python
from clam.utils import save_results

# Legacy usage (still works)
save_results(results_list, output_dir, dataset_name)

# Optionally use unified manager
save_results(results_list, output_dir, dataset_name, use_unified_manager=True)
```

## Directory Structure

The unified system organizes results in a hierarchical structure:

```
<base_dir>/results/
├── tabular/
│   ├── adult/
│   │   ├── tabllm/
│   │   │   ├── results.json           # Main results file
│   │   │   ├── metadata.json          # Experiment metadata
│   │   │   ├── metrics_summary.json   # Quick metrics access
│   │   │   └── artifacts/              # Additional files
│   │   ├── jolt/
│   │   └── clam_tsne/
│   ├── diabetes/
│   └── wine/
├── vision/
│   ├── cifar10/
│   │   ├── clam_tsne/
│   │   │   ├── results.json
│   │   │   ├── metadata.json
│   │   │   └── artifacts/
│   │   │       ├── visualizations/
│   │   │       └── vlm_responses.json
│   │   └── dinov2_linear/
│   └── imagenet/
└── audio/
    ├── esc50/
    │   ├── clam_audio/
    │   └── whisper_knn/
    └── ravdess/
```

## Data Structures

### EvaluationResults

Standardized structure for evaluation metrics and outputs:

```python
@dataclass
class EvaluationResults:
    # Core metrics
    accuracy: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    r2_score: Optional[float] = None  # For regression
    mae: Optional[float] = None
    rmse: Optional[float] = None
    
    # Classification metrics
    precision_macro: Optional[float] = None
    recall_macro: Optional[float] = None
    f1_macro: Optional[float] = None
    classification_report: Optional[Dict[str, Any]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    
    # Performance metrics
    completion_rate: Optional[float] = None
    total_prediction_time: Optional[float] = None
    prediction_time_per_sample: Optional[float] = None
    
    # Outputs
    predictions: Optional[List[Any]] = None
    raw_responses: Optional[List[str]] = None  # For LLM models
    visualization_paths: Optional[List[str]] = None
    
    # Status
    status: str = "completed"  # "completed", "failed", "partial"
    error_message: Optional[str] = None
```

### ExperimentMetadata

Comprehensive metadata for reproducibility:

```python
@dataclass 
class ExperimentMetadata:
    model_name: str
    dataset_id: str
    modality: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    # Dataset information
    num_samples_train: Optional[int] = None
    num_samples_test: Optional[int] = None
    num_features: Optional[int] = None
    num_classes: Optional[int] = None
    class_names: Optional[List[str]] = None
    
    # Model configuration
    model_config: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    
    # Experiment setup
    random_seed: Optional[int] = None
    k_shot: Optional[int] = None
    device: Optional[str] = None
    training_time_seconds: Optional[float] = None
```

### ResultsArtifacts

Container for additional experiment artifacts:

```python
@dataclass
class ResultsArtifacts:
    plots: Optional[List[str]] = None
    visualizations: Optional[Dict[str, str]] = None
    raw_outputs: Optional[str] = None
    predictions_csv: Optional[str] = None
    
    # Modality-specific
    audio_files: Optional[List[str]] = None
    image_files: Optional[List[str]] = None
```

## Migrating Existing Results

### Automatic Migration

Use the CLI tool for batch migration:

```bash
# Dry run to see what would be migrated
python scripts/results_cli.py migrate /path/to/legacy/results --dry-run

# Actually migrate
python scripts/results_cli.py migrate /path/to/legacy/results --no-dry-run
```

### Programmatic Migration

```python
from clam.utils import migrate_legacy_results

# Migrate legacy results
stats = migrate_legacy_results(
    source_dir="/path/to/legacy/results",
    pattern="*_results.json",
    dry_run=False
)

print(f"Migrated {stats['successful']}/{stats['total_files']} files")
```

### Manual Conversion

For custom result formats, use the format detector:

```python
from clam.utils import ResultsFormatDetector, ResultsMigrator

detector = ResultsFormatDetector()
migrator = ResultsMigrator()

# Detect format
with open('result.json', 'r') as f:
    result_dict = json.load(f)

format_type = detector.detect_format(result_dict)
print(f"Detected format: {format_type}")

# Migrate single file
success = migrator.migrate_file(
    'result.json',
    target_model_name='custom_model',
    target_dataset_id='custom_dataset',
    target_modality='tabular'
)
```

## Integration with Evaluation Scripts

### Option 1: Direct Integration

Update existing scripts to use the unified manager:

```python
def evaluate_model(model_name, dataset, args):
    # ... existing evaluation code ...
    
    # Convert results to standardized format
    results = EvaluationResults(
        accuracy=accuracy,
        f1_macro=f1_score,
        status="completed"
    )
    
    metadata = ExperimentMetadata(
        model_name=model_name,
        dataset_id=dataset['id'],
        modality="tabular",
        k_shot=args.k_shot,
        random_seed=args.seed
    )
    
    # Save using unified manager
    results_manager = get_results_manager()
    experiment_dir = results_manager.save_evaluation_results(
        model_name=model_name,
        dataset_id=dataset['id'],
        modality="tabular",
        results=results,
        experiment_metadata=metadata
    )
    
    return experiment_dir
```

### Option 2: Gradual Transition

Add an option to use the unified manager:

```python
# Add argument
parser.add_argument("--use_unified_results", action="store_true")

# Use conditionally
if args.use_unified_results:
    save_results_unified(results, dataset_name, model_name, "tabular")
else:
    save_results(results, output_dir, dataset_name)  # Legacy
```

### Option 3: Adapter Functions

Use adapter functions for different script patterns:

```python
from clam.utils import create_migration_adapters

adapters = create_migration_adapters()

# Adapt tabular LLM results
adapted_result = adapters['tabular_llm'](
    model_name="tabllm",
    dataset_name="adult", 
    results=raw_results,
    args=args
)
```

## CLI Tools

The system includes command-line tools for management:

### List Experiments

```bash
# List all experiments
python scripts/results_cli.py list

# Filter by modality
python scripts/results_cli.py list --modality tabular

# Filter by dataset
python scripts/results_cli.py list --dataset-id adult
```

### Generate Reports

```bash
# Generate summary report
python scripts/results_cli.py report

# Save detailed report
python scripts/results_cli.py report --output summary.json

# Filter by modality
python scripts/results_cli.py report --modality vision
```

### Cleanup

```bash
# Dry run cleanup (show what would be deleted)
python scripts/results_cli.py cleanup --days 30 --dry-run

# Actually delete old results
python scripts/results_cli.py cleanup --days 30 --no-dry-run
```

### Validation

```bash
# Validate a result file
python scripts/results_cli.py validate result.json
```

### System Information

```bash
# Show system info
python scripts/results_cli.py info
```

## Advanced Features

### Artifact Management

Save additional files with experiments:

```python
# Save artifacts separately
artifact_paths = results_manager.save_artifacts(
    model_name="clam_tsne",
    dataset_id="cifar10",
    modality="vision",
    artifacts={
        "tsne_plot": "/tmp/tsne_visualization.png",
        "vlm_responses": "/tmp/responses.json",
        "confusion_matrix": "/tmp/cm.png"
    },
    copy_files=True
)
```

### Summary Reports

Generate comprehensive reports:

```python
# Create summary report
report = results_manager.create_summary_report(
    modality="tabular",
    output_file="tabular_summary.json"
)

print(f"Total experiments: {report['summary']['total_experiments']}")
print(f"Best accuracy: {max(exp['accuracy'] for exp in report['experiments'] if exp.get('accuracy'))}")
```

### Cleanup Management

Automatically manage old results:

```python
# Clean up results older than 30 days
stats = results_manager.cleanup_old_results(
    days_old=30,
    dry_run=False
)

print(f"Cleaned up {stats['deleted']} experiments")
print(f"Freed {stats['freed_size_mb']:.1f} MB")
```

## Configuration

The system uses the existing resource manager configuration:

```python
from clam.utils import ResourceConfig, get_results_manager

# Custom configuration
config = ResourceConfig(
    base_dir="/custom/clam/dir",
    cache_dir="/custom/cache"
)

# Use with results manager
results_manager = get_results_manager()
```

Environment variables:
- `CLAM_BASE_DIR`: Base directory for all CLAM data
- `CLAM_CACHE_DIR`: Cache directory
- `CLAM_DISABLE_REGISTRY`: Disable dataset registry

## Best Practices

### 1. Consistent Naming

Use consistent naming conventions:
- **Model names**: Use snake_case (e.g., `tabllm`, `clam_tsne`, `dinov2_linear`)
- **Dataset IDs**: Use original identifiers when possible (e.g., OpenML task IDs)
- **Modalities**: Use standard names (`tabular`, `vision`, `audio`)

### 2. Comprehensive Metadata

Include as much metadata as possible:

```python
metadata = ExperimentMetadata(
    model_name="tabllm",
    dataset_id="adult",
    modality="tabular",
    num_samples_train=32561,
    num_samples_test=16281,
    num_features=14,
    num_classes=2,
    class_names=["<=50K", ">50K"],
    k_shot=5,
    random_seed=42,
    model_config={
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "temperature": 0.1,
        "max_tokens": 100
    },
    hyperparameters={
        "num_few_shot_examples": 32,
        "balanced_few_shot": True
    },
    notes="Experiment to test TabLLM performance on income prediction"
)
```

### 3. Error Handling

Always handle errors gracefully:

```python
try:
    experiment_dir = results_manager.save_evaluation_results(...)
    logger.info(f"Results saved to {experiment_dir}")
except Exception as e:
    logger.error(f"Failed to save results: {e}")
    # Fall back to legacy method
    save_results(results, output_dir, dataset_name)
```

### 4. Regular Cleanup

Set up regular cleanup of old results:

```python
# In your evaluation scripts
if datetime.datetime.now().day == 1:  # Monthly cleanup
    stats = results_manager.cleanup_old_results(days_old=90)
    logger.info(f"Monthly cleanup: freed {stats['freed_size_mb']:.1f} MB")
```

### 5. Validation

Validate results before saving:

```python
from clam.utils import validate_result_file

# Validate before using
validation = validate_result_file("result.json")
if validation['status'] == 'valid':
    # Process the result
    pass
else:
    logger.error(f"Invalid result file: {validation['error']}")
```

## Troubleshooting

### Common Issues

1. **Permission Errors**
   ```
   Solution: Check directory permissions for CLAM base directory
   ```

2. **Migration Failures**
   ```
   Solution: Use dry_run=True first, check file formats
   ```

3. **Large Result Files**
   ```
   Solution: Use artifacts for large files, enable compression
   ```

4. **Concurrent Access**
   ```
   Solution: The system handles concurrent access automatically
   ```

### Debug Information

Enable debug logging:

```python
import logging
logging.getLogger('clam.utils.results_manager').setLevel(logging.DEBUG)
```

Check system status:

```python
results_manager = get_results_manager()
status = results_manager.get_registry_status() if hasattr(results_manager, 'get_registry_status') else {}
print(f"System status: {status}")
```

## Migration from Legacy System

### Step 1: Identify Legacy Results

```bash
find . -name "*_results.json" -type f
```

### Step 2: Validate Legacy Format

```python
from clam.utils import validate_result_file

for file_path in legacy_files:
    validation = validate_result_file(file_path)
    print(f"{file_path}: {validation['status']}")
```

### Step 3: Migrate

```bash
python scripts/results_cli.py migrate ./legacy_results --no-dry-run
```

### Step 4: Verify Migration

```bash
python scripts/results_cli.py list
python scripts/results_cli.py report --output migration_report.json
```

### Step 5: Update Scripts

Add unified results support to evaluation scripts using the examples provided.

## API Reference

### Core Classes

- `ResultsManager`: Main interface for results management
- `EvaluationResults`: Standardized evaluation metrics
- `ExperimentMetadata`: Comprehensive experiment metadata
- `ResultsArtifacts`: Container for additional files

### Functions

- `get_results_manager()`: Get global results manager instance
- `save_results_unified()`: Unified save function with backward compatibility
- `migrate_legacy_results()`: Migrate legacy result directories
- `validate_result_file()`: Validate individual result files

### CLI Commands

- `results_cli.py list`: List experiments
- `results_cli.py migrate`: Migrate legacy results
- `results_cli.py report`: Generate reports
- `results_cli.py cleanup`: Clean up old results
- `results_cli.py validate`: Validate result files
- `results_cli.py info`: Show system information

For more examples and detailed API documentation, see the demonstration scripts in `scripts/results_manager_demo.py`.