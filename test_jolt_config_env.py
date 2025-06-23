#!/usr/bin/env python3
"""
Test JOLT config loading with custom environment variables to reproduce the remote server issue.
"""

import os
import sys
import tempfile
import shutil
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_jolt_config_with_custom_env():
    """Test JOLT config generation and loading with custom CLAM_BASE_DIR."""
    print("=== Testing JOLT Config with Custom Environment ===")
    
    # Create a temporary directory to simulate the remote server setup
    temp_base = Path(tempfile.mkdtemp()) / "clam"
    temp_base.mkdir(parents=True, exist_ok=True)
    
    try:
        # Set environment variable like on remote server
        original_base_dir = os.environ.get('CLAM_BASE_DIR')
        os.environ['CLAM_BASE_DIR'] = str(temp_base)
        
        print(f"Set CLAM_BASE_DIR to: {temp_base}")
        
        # Step 1: Test resource manager path resolution
        from clam.utils.resource_manager import get_resource_manager
        rm = get_resource_manager()
        
        print(f"\nResource Manager Paths:")
        print(f"  Base dir: {rm.path_resolver.get_base_dir()}")
        print(f"  Configs dir: {rm.path_resolver.get_configs_dir()}")
        print(f"  JOLT configs dir: {rm.path_resolver.get_configs_dir() / 'jolt'}")
        
        # Step 2: Create a test JOLT config manually
        jolt_configs_dir = rm.path_resolver.get_configs_dir() / 'jolt'
        jolt_configs_dir.mkdir(parents=True, exist_ok=True)
        
        test_task_id = "23"
        test_config = {
            "task_id": test_task_id,
            "dataset_name": "cmc",
            "task_prefix": "Test contraceptive method choice prediction",
            "column_descriptions": {
                "feature_1": "Wife's age",
                "feature_2": "Wife's education"
            },
            "class_names": ["1", "2", "3"],
            "class_description": "Classes: 1, 2, 3",
            "num_features": 2,
            "num_classes": 3
        }
        
        config_filename = f"jolt_config_task_{test_task_id}.json"
        config_path = jolt_configs_dir / config_filename
        
        with open(config_path, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        print(f"\nCreated test config at: {config_path}")
        print(f"Config file exists: {config_path.exists()}")
        
        # Step 3: Test config loading using the same logic as JOLT wrapper
        print(f"\nTesting config loading...")
        
        # Simulate the dataset dict that JOLT wrapper receives
        test_dataset = {
            'id': test_task_id,
            'name': 'cmc',
            'data_source': 'openml'
        }
        
        # Test the exact same logic as in the JOLT wrapper
        jolt_config = None
        
        # First try the task-based naming convention (newer format)
        task_config_filename = f"jolt_config_task_{test_dataset['id']}.json"
        jolt_config_path = rm.path_resolver.get_config_path('jolt', task_config_filename)
        
        print(f"  Looking for task config: {task_config_filename}")
        print(f"  Task config path: {jolt_config_path}")
        print(f"  Task config exists: {jolt_config_path.exists() if jolt_config_path else False}")
        
        if jolt_config_path and jolt_config_path.exists():
            with open(jolt_config_path, 'r') as f:
                jolt_config = json.load(f)
            print(f"  ✅ Successfully loaded config: {jolt_config.get('task_prefix', 'No prefix')}")
            config_loaded = True
        else:
            print(f"  ❌ Task-based config not found")
            config_loaded = False
        
        # Step 4: Test the JOLT wrapper import and path resolution
        print(f"\nTesting JOLT wrapper config loading...")
        
        # Import the JOLT wrapper (this will use the same environment)
        sys.path.insert(0, str(Path(__file__).parent / "examples" / "tabular" / "llm_baselines" / "jolt"))
        
        # Simulate just the config loading part of the wrapper
        try:
            from clam.utils.resource_manager import get_resource_manager as get_rm_2
            rm2 = get_rm_2()
            
            print(f"  JOLT wrapper resource manager paths:")
            print(f"    Base dir: {rm2.path_resolver.get_base_dir()}")
            print(f"    Configs dir: {rm2.path_resolver.get_configs_dir()}")
            
            # Test config loading exactly as in wrapper
            task_config_filename = f"jolt_config_task_{test_dataset['id']}.json"
            jolt_config_path = rm2.path_resolver.get_config_path('jolt', task_config_filename)
            
            print(f"    Looking for: {task_config_filename}")
            print(f"    Path returned: {jolt_config_path}")
            print(f"    File exists: {jolt_config_path.exists() if jolt_config_path else False}")
            
            if jolt_config_path and jolt_config_path.exists():
                with open(jolt_config_path, 'r') as f:
                    wrapper_config = json.load(f)
                print(f"    ✅ Wrapper successfully loaded config")
                wrapper_loaded = True
            else:
                print(f"    ❌ Wrapper failed to load config")
                wrapper_loaded = False
                
        except Exception as e:
            print(f"    ❌ Error testing wrapper: {e}")
            wrapper_loaded = False
        
        # Step 5: Summary
        print(f"\n=== SUMMARY ===")
        print(f"Environment CLAM_BASE_DIR: {os.environ.get('CLAM_BASE_DIR')}")
        print(f"Config created at: {config_path}")
        print(f"Direct config loading: {'✅ SUCCESS' if config_loaded else '❌ FAILED'}")
        print(f"Wrapper config loading: {'✅ SUCCESS' if wrapper_loaded else '❌ FAILED'}")
        
        if config_loaded and wrapper_loaded:
            print(f"🎉 Both direct and wrapper config loading work with custom CLAM_BASE_DIR")
            return True
        else:
            print(f"💥 Config loading failed - this reproduces the remote server issue")
            return False
            
    finally:
        # Cleanup
        if original_base_dir is not None:
            os.environ['CLAM_BASE_DIR'] = original_base_dir
        elif 'CLAM_BASE_DIR' in os.environ:
            del os.environ['CLAM_BASE_DIR']
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_base.parent)
        except Exception as e:
            print(f"Warning: Could not clean up temp dir {temp_base.parent}: {e}")

if __name__ == "__main__":
    success = test_jolt_config_with_custom_env()
    sys.exit(0 if success else 1)