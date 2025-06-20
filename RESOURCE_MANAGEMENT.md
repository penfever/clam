# CLAM Resource Management System

This document describes the new robust resource management system implemented in CLAM v2.0, which replaces the previous fragile path construction patterns with a centralized, organized approach to dataset management, caching, and configuration handling.

## Overview

The resource management system provides:
- **Centralized Resource Management**: Single point of control for all CLAM resources
- **Isolated Dataset Workspaces**: Each dataset gets its own managed directory
- **Robust Caching**: Smart caching with size limits and automatic cleanup
- **Configuration Management**: Organized storage and discovery of metadata configs
- **Migration Support**: Seamless migration from legacy scattered resources
- **Environment Configuration**: Configurable via environment variables

## Architecture

### Components

1. **ResourceManager**: Main coordinator that ties everything together
2. **PathResolver**: Handles all path resolution and directory management
3. **CacheManager**: Manages caching with size limits and cleanup
4. **DatasetRegistry**: Tracks dataset metadata and locations
5. **MigrationManager**: Migrates existing resources to new structure

### Directory Structure

```
~/.clam/                          # Base directory (configurable)
├── registry.json                # Dataset registry
├── datasets/                    # Dataset workspaces
│   ├── dataset_adult_1590/      # OpenML dataset workspace
│   │   ├── metadata.json        # Dataset-specific metadata
│   │   ├── embeddings/          # Cached embeddings
│   │   └── cache/               # Dataset-specific cache
│   └── dataset_esc50/           # Audio dataset workspace
├── configs/                     # Managed configurations
│   ├── jolt/                   # JOLT configurations
│   │   ├── jolt_config_adult.json
│   │   └── jolt_config_diabetes.json
│   ├── tabllm/                 # TabLLM configurations
│   │   ├── templates_adult.yaml
│   │   ├── templates_diabetes.yaml
│   │   └── openml_task_mapping.json
│   └── cc18_semantic/          # CC18 semantic metadata
│       ├── 1590.json           # OpenML task metadata
│       └── 37.json
└── cache/                      # Global cache
    ├── embeddings/             # Embedding cache
    ├── system/                 # System cache (failed datasets, etc.)
    └── temp/                   # Temporary cache
```

## Usage

### Basic Usage

```python
from clam.utils.resource_manager import get_resource_manager

# Get the global resource manager
rm = get_resource_manager()

# Get dataset workspace
workspace = rm.get_dataset_workspace("adult")

# Find CSV files robustly
csv_path = rm.find_csv_file("adult", additional_search_dirs=["./data"])

# Load configuration
config_path = rm.path_resolver.get_config_path("jolt", "adult")
```

### Environment Configuration

Set environment variables to customize behavior:

```bash
export CLAM_BASE_DIR="/custom/clam/directory"
export CLAM_CACHE_DIR="/fast/ssd/cache"
export CLAM_MAX_CACHE_GB="20.0"
export CLAM_ENABLE_CACHING="true"
```

### Dataset Registration

```python
from clam.utils.resource_manager import DatasetMetadata

# Register a new dataset
metadata = DatasetMetadata(
    id="my_dataset",
    name="My Custom Dataset",
    source_type="csv",
    file_path="/path/to/data.csv",
    num_samples=1000,
    num_features=10,
    num_classes=3,
    task_type="classification"
)

rm.dataset_registry.register_dataset(metadata)
```

### Caching

```python
# Generate cache key
cache_key = rm.cache_manager.get_cache_key(
    dataset="adult",
    model="tabpfn",
    embedding_size=1000
)

# Save to cache
embeddings_data = {
    "train_embeddings": train_emb,
    "val_embeddings": val_emb,
    "test_embeddings": test_emb
}

success = rm.cache_manager.save_to_cache(
    "embeddings", cache_key, embeddings_data, ".npz"
)

# Load from cache
cached_data = rm.cache_manager.load_from_cache(
    "embeddings", cache_key, ".npz"
)
```

## Migration

The system includes automatic migration from the legacy scattered resource structure.

### Running Migration

```bash
# Dry run to see what would be migrated
python scripts/resource_management_demo.py --migrate --dry-run

# Live migration
python scripts/resource_management_demo.py --migrate

# Or use the migration utilities directly
python -m clam.utils.migration_utils --live --verbose
```

### What Gets Migrated

1. **Failed Dataset Cache**: `~/.clam_failed_datasets.json` → managed cache
2. **Embedding Caches**: `*tabpfn_embeddings*.npz` files → managed cache
3. **JOLT Configs**: `examples/tabular/llm_baselines/jolt/jolt_config_*.json` → configs/jolt/
4. **TabLLM Configs**: `examples/tabular/llm_baselines/tabllm_like/templates_*.yaml` → configs/tabllm/
5. **CC18 Semantic**: `data/cc18_semantic/*.json` → configs/cc18_semantic/

## Backward Compatibility

The system maintains backward compatibility through:

1. **Gradual Migration**: Existing code continues to work while new system is adopted
2. **Fallback Mechanisms**: If managed resources aren't found, falls back to legacy paths
3. **Legacy Support**: All existing APIs continue to work unchanged

## Integration Points

### Dataset Loading (`clam/data/dataset.py`)
- Automatically registers successfully loaded datasets
- Uses managed cache for failed dataset tracking
- Maintains compatibility with existing dataset loading code

### Embedding Caching (`clam/data/embeddings.py`)
- Uses managed cache for TabPFN embeddings when available
- Falls back to legacy caching if resource manager unavailable
- Automatic cache key generation and collision handling

### Configuration Loading
- **JOLT**: `examples/tabular/llm_baselines/jolt/official_jolt_wrapper.py`
- **TabLLM**: `examples/tabular/llm_baselines/tabllm_baseline.py`
- Both use managed config paths with legacy fallbacks

### CSV File Discovery (`clam/data/csv_utils.py`)
- Enhanced `find_csv_with_fallbacks()` uses resource manager
- Maintains compatibility with existing search patterns
- More robust path resolution

## Benefits

### For Users
- **Cleaner Workspace**: All CLAM resources organized in one place
- **Better Performance**: Smart caching reduces recomputation
- **Easier Debugging**: Clear resource locations and metadata
- **Cross-Session Sharing**: Cached resources persist and can be shared

### For Developers
- **Reduced Path Fragility**: No more hardcoded relative paths
- **Better Resource Isolation**: Datasets don't interfere with each other
- **Easier Testing**: Configurable base directories for test isolation
- **Consistent APIs**: Single interface for all resource operations

### For System Administrators
- **Configurable Storage**: Can redirect to fast SSDs, network storage, etc.
- **Size Management**: Automatic cache size limits and cleanup
- **Monitoring**: Clear visibility into resource usage
- **Backup-Friendly**: Single directory contains all important resources

## Testing

Run the comprehensive test suite:

```bash
python tests/test_resource_management.py
```

Or use the demonstration script:

```bash
python scripts/resource_management_demo.py --verbose
```

## Advanced Configuration

### Custom Resource Manager

```python
from clam.utils.resource_manager import ResourceConfig, initialize_resource_manager

config = ResourceConfig(
    base_dir="/custom/clam/dir",
    cache_dir="/fast/ssd/cache",
    max_cache_size_gb=50.0,
    enable_caching=True
)

rm = initialize_resource_manager(config)
```

### Cache Management

```python
# Check cache size
cache_dir = rm.path_resolver.get_cache_dir()
total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())

# Force cache cleanup
rm.cache_manager._cleanup_cache_if_needed()

# Disable caching for specific operations
rm.config.enable_caching = False
```

### Dataset Discovery

```python
from clam.utils.migration_utils import MigrationManager

migration_manager = MigrationManager(rm)

# Discover CSV files in various directories
discovered = migration_manager.discover_existing_datasets([
    "./data",
    "./datasets", 
    "/shared/datasets"
])

# Register discovered datasets
count = migration_manager.register_discovered_datasets(discovered)
print(f"Registered {count} datasets")
```

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure write access to base directory
2. **Disk Space**: Monitor cache size limits
3. **Migration Issues**: Run dry-run first to identify problems
4. **Path Issues**: Check environment variables and config

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose logging for resource manager
logging.getLogger('clam.utils.resource_manager').setLevel(logging.DEBUG)
```

### Reset Everything

```bash
# Remove all managed resources (DESTRUCTIVE!)
rm -rf ~/.clam

# Or reset just the cache
rm -rf ~/.clam/cache
```

## Contributing

When adding new resource types:

1. Add path resolution logic to `PathResolver`
2. Add migration logic to `MigrationManager`
3. Update configuration schemas if needed
4. Add tests to `test_resource_management.py`
5. Update this documentation

## Future Enhancements

Planned improvements:
- **Network Storage Support**: Remote cache backends
- **Compression**: Automatic compression for large caches
- **Replication**: Multi-location resource management
- **Analytics**: Resource usage tracking and optimization
- **Integration**: Better IDE and notebook integration