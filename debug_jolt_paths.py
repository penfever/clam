#!/usr/bin/env python3
"""
Debug script to check JOLT config paths on remote server.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from clam.utils.resource_manager import get_resource_manager

def debug_jolt_paths():
    """Debug JOLT config path resolution."""
    print("=== JOLT Config Path Debug ===")
    
    # Print environment variables
    env_vars = ['CLAM_BASE_DIR', 'CLAM_CONFIGS_DIR', 'CLAM_CACHE_DIR', 'CLAM_DATASETS_DIR']
    print("\nEnvironment Variables:")
    for var in env_vars:
        value = os.environ.get(var, 'NOT_SET')
        print(f"  {var}: {value}")
    
    # Get resource manager
    rm = get_resource_manager()
    
    print("\nResource Manager Paths:")
    print(f"  Base dir: {rm.path_resolver.get_base_dir()}")
    print(f"  Configs dir: {rm.path_resolver.get_configs_dir()}")
    print(f"  JOLT configs dir: {rm.path_resolver.get_configs_dir() / 'jolt'}")
    
    # Check if JOLT configs directory exists
    jolt_dir = rm.path_resolver.get_configs_dir() / 'jolt'
    print(f"\nJOLT Directory Check:")
    print(f"  JOLT dir exists: {jolt_dir.exists()}")
    
    if jolt_dir.exists():
        # List some files
        try:
            all_files = list(jolt_dir.glob("*.json"))
            print(f"  Total JSON files: {len(all_files)}")
            
            task_files = [f for f in all_files if f.name.startswith('jolt_config_task_')]
            print(f"  Task-based config files: {len(task_files)}")
            
            if task_files:
                print(f"  Sample task files: {[f.name for f in task_files[:5]]}")
        except Exception as e:
            print(f"  Error listing files: {e}")
    
    # Test config loading for specific tasks
    test_tasks = ['23', '3', '53']
    print(f"\nConfig Loading Test:")
    for task_id in test_tasks:
        print(f"\n  Task {task_id}:")
        
        # Test task-based naming
        task_filename = f"jolt_config_task_{task_id}.json"
        task_path = rm.path_resolver.get_config_path('jolt', task_filename)
        
        print(f"    Looking for: {task_filename}")
        print(f"    Path returned: {task_path}")
        print(f"    File exists: {task_path.exists() if task_path else False}")
        
        if task_path and task_path.exists():
            print(f"    ✅ Config found and loadable")
        else:
            print(f"    ❌ Config not found")
            
            # Check if file exists in expected location
            expected_path = jolt_dir / task_filename
            print(f"    Expected path: {expected_path}")
            print(f"    Expected exists: {expected_path.exists()}")

if __name__ == "__main__":
    debug_jolt_paths()