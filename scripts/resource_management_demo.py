#!/usr/bin/env python3
"""
Demonstration script for the new CLAM resource management system.

This script shows how to use the resource manager for dataset management,
configuration loading, and caching.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add the parent directory to the path so we can import clam modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from clam.utils.resource_manager import (
    get_resource_manager, 
    initialize_resource_manager,
    ResourceConfig,
    DatasetMetadata
)
from clam.utils.migration_utils import run_migration


def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demo_resource_manager():
    """Demonstrate basic resource manager functionality."""
    print("=== CLAM Resource Manager Demo ===\n")
    
    # Initialize the resource manager
    print("1. Initializing resource manager...")
    rm = get_resource_manager()
    
    print(f"   Base directory: {rm.path_resolver.base_dir}")
    print(f"   Datasets directory: {rm.path_resolver.get_datasets_dir()}")
    print(f"   Cache directory: {rm.path_resolver.get_cache_dir()}")
    print(f"   Configs directory: {rm.path_resolver.get_configs_dir()}")
    
    # Show dataset registry
    print("\n2. Current dataset registry:")
    datasets = rm.dataset_registry.list_datasets()
    if datasets:
        for dataset in datasets[:5]:  # Show first 5
            print(f"   - {dataset.name} (ID: {dataset.id}, Type: {dataset.source_type})")
        if len(datasets) > 5:
            print(f"   ... and {len(datasets) - 5} more datasets")
    else:
        print("   No datasets registered yet")
    
    # Demonstrate config path resolution
    print("\n3. Config path resolution:")
    for config_type in ['jolt', 'tabllm', 'cc18_semantic']:
        config_path = rm.path_resolver.get_config_path(config_type, 'adult')
        if config_path:
            exists = "✓" if config_path.exists() else "✗"
            print(f"   {config_type}: {exists} {config_path}")
        else:
            print(f"   {config_type}: ✗ Not configured")
    
    # Show cache info
    print("\n4. Cache information:")
    cache_dir = rm.path_resolver.get_cache_dir()
    if cache_dir.exists():
        cache_files = list(cache_dir.rglob('*'))
        cache_count = len([f for f in cache_files if f.is_file()])
        total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
        print(f"   Cache files: {cache_count}")
        print(f"   Total size: {total_size / 1024 / 1024:.2f} MB")
    else:
        print("   Cache directory not yet created")
    
    return rm


def demo_dataset_registration(rm):
    """Demonstrate dataset registration."""
    print("\n=== Dataset Registration Demo ===\n")
    
    # Register a sample dataset
    print("1. Registering a sample dataset...")
    sample_metadata = DatasetMetadata(
        id="demo_dataset",
        name="Demo Dataset",
        source_type="csv",
        file_path="/path/to/demo.csv",
        num_samples=1000,
        num_features=10,
        num_classes=3,
        task_type="classification"
    )
    
    success = rm.dataset_registry.register_dataset(sample_metadata)
    if success:
        print("   ✓ Successfully registered demo dataset")
    else:
        print("   ✗ Failed to register demo dataset")
    
    # Retrieve the dataset
    print("\n2. Retrieving registered dataset...")
    retrieved = rm.dataset_registry.get_dataset_metadata("demo_dataset")
    if retrieved:
        print(f"   ✓ Found dataset: {retrieved.name}")
        print(f"     Source: {retrieved.source_type}")
        print(f"     Samples: {retrieved.num_samples}")
        print(f"     Features: {retrieved.num_features}")
        print(f"     Classes: {retrieved.num_classes}")
        print(f"     Task: {retrieved.task_type}")
    else:
        print("   ✗ Could not retrieve dataset")
    
    return retrieved is not None


def demo_caching(rm):
    """Demonstrate caching functionality."""
    print("\n=== Caching Demo ===\n")
    
    # Test data to cache
    import numpy as np
    test_data = {
        "embeddings": np.random.randn(100, 50),
        "labels": np.random.randint(0, 3, 100),
        "metadata": {"created_by": "demo", "version": "1.0"}
    }
    
    print("1. Saving data to cache...")
    cache_key = rm.cache_manager.get_cache_key(
        dataset="demo",
        model="tabpfn",
        version="1.0"
    )
    
    success = rm.cache_manager.save_to_cache(
        'embeddings', cache_key, test_data, '.npz'
    )
    
    if success:
        print(f"   ✓ Saved to cache with key: {cache_key}")
    else:
        print("   ✗ Failed to save to cache")
        return False
    
    print("\n2. Loading data from cache...")
    loaded_data = rm.cache_manager.load_from_cache(
        'embeddings', cache_key, '.npz'
    )
    
    if loaded_data:
        print("   ✓ Successfully loaded from cache")
        print(f"     Embeddings shape: {loaded_data['embeddings'].shape}")
        print(f"     Labels shape: {loaded_data['labels'].shape}")
        print(f"     Metadata: {loaded_data['metadata']}")
        return True
    else:
        print("   ✗ Failed to load from cache")
        return False


def demo_csv_finding(rm):
    """Demonstrate CSV file finding."""
    print("\n=== CSV File Finding Demo ===\n")
    
    # Try to find some common dataset CSV files
    test_datasets = ["adult", "diabetes", "titanic", "iris"]
    
    print("1. Searching for CSV files...")
    for dataset_id in test_datasets:
        csv_path = rm.find_csv_file(dataset_id)
        if csv_path:
            print(f"   ✓ {dataset_id}: {csv_path}")
        else:
            print(f"   ✗ {dataset_id}: Not found")


def main():
    parser = argparse.ArgumentParser(description="CLAM Resource Management Demo")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--migrate", action="store_true", help="Run migration from legacy system")
    parser.add_argument("--dry-run", action="store_true", help="Dry run migration (no actual changes)")
    parser.add_argument("--custom-base-dir", type=str, help="Use custom base directory")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    print("CLAM Resource Management System Demo")
    print("=====================================")
    
    # Initialize with custom config if requested
    if args.custom_base_dir:
        config = ResourceConfig(base_dir=args.custom_base_dir)
        rm = initialize_resource_manager(config)
        print(f"\nUsing custom base directory: {args.custom_base_dir}")
    else:
        rm = get_resource_manager()
    
    # Run migration if requested
    if args.migrate:
        print("\n=== Running Migration ===\n")
        results = run_migration(dry_run=args.dry_run)
        
        print("Migration results:")
        for key, count in results.items():
            if count > 0:
                print(f"  {key}: {count}")
        print()
    
    # Run demos
    try:
        demo_resource_manager()
        demo_dataset_registration(rm)
        demo_caching(rm)
        demo_csv_finding(rm)
        
        print("\n=== Demo Complete ===")
        print("The resource management system is working correctly!")
        print(f"All resources are organized under: {rm.path_resolver.base_dir}")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())