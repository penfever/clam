#!/usr/bin/env python3
"""
Test JOLT config loading with exact remote server paths and scenarios.
"""

import os
import sys
import tempfile
import shutil
import json
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_exact_remote_server_scenario():
    """Test the exact scenario from the remote server."""
    print("=== Testing Exact Remote Server Scenario ===")
    
    # Create temporary directories that match remote server structure
    temp_dir = Path(tempfile.mkdtemp())
    
    # Simulate remote server paths
    clam_base = temp_dir / "home" / "benjamin" / "clam"
    clam_configs = clam_base / "configs" / "jolt"
    clam_configs.mkdir(parents=True, exist_ok=True)
    
    try:
        # Set environment exactly like remote server
        original_env = {}
        remote_env = {
            'CLAM_BASE_DIR': str(clam_base),
            'CLAM_DATASETS_DIR': str(temp_dir / "data" / "benjamin" / "clam_datasets"),
            'CLAM_CACHE_DIR': str(temp_dir / "data" / "benjamin" / "clam_cache")
        }
        
        for key, value in remote_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        print(f"Set remote server environment:")
        for key, value in remote_env.items():
            print(f"  {key}: {value}")
        
        # Step 1: Simulate config generation (like synthesize_jolt_data.py)
        print(f"\n=== Step 1: Simulating Config Generation ===")
        
        from clam.utils.resource_manager import get_resource_manager
        rm = get_resource_manager()
        
        generation_output_dir = rm.path_resolver.get_configs_dir() / 'jolt'
        print(f"Config generation would save to: {generation_output_dir}")
        
        # Create some test configs
        test_configs = [
            {"task_id": "23", "name": "cmc"},
            {"task_id": "53", "name": "vehicle"},
            {"task_id": "3", "name": "kr-vs-kp"}
        ]
        
        for config_info in test_configs:
            config_data = {
                "task_id": config_info["task_id"],
                "dataset_name": config_info["name"],
                "task_prefix": f"Test task {config_info['task_id']} prediction",
                "column_descriptions": {"feature_1": "Test feature"},
                "class_names": ["Class_0", "Class_1"],
                "class_description": "Classes: Class_0, Class_1",
                "num_features": 1,
                "num_classes": 2
            }
            
            config_filename = f"jolt_config_task_{config_info['task_id']}.json"
            config_path = generation_output_dir / config_filename
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"  Created: {config_filename}")
        
        print(f"  Total configs created: {len(list(generation_output_dir.glob('*.json')))}")
        
        # Step 2: Test config loading in separate process (simulating evaluation)
        print(f"\n=== Step 2: Testing Config Loading (Simulating Evaluation) ===")
        
        # Reload resource manager to simulate fresh process
        import importlib
        import clam.utils.resource_manager
        importlib.reload(clam.utils.resource_manager)
        from clam.utils.resource_manager import get_resource_manager
        
        rm2 = get_resource_manager()
        
        print(f"Evaluation process resource manager paths:")
        print(f"  Base dir: {rm2.path_resolver.get_base_dir()}")
        print(f"  Configs dir: {rm2.path_resolver.get_configs_dir()}")
        
        # Test loading each config
        for config_info in test_configs:
            task_id = config_info["task_id"]
            dataset_name = config_info["name"]
            
            print(f"\n  Testing task {task_id} ({dataset_name}):")
            
            # Test task-based loading
            task_config_filename = f"jolt_config_task_{task_id}.json"
            task_config_path = rm2.path_resolver.get_config_path('jolt', task_config_filename)
            
            print(f"    Task filename: {task_config_filename}")
            print(f"    Task path: {task_config_path}")
            print(f"    Task exists: {task_config_path.exists() if task_config_path else False}")
            
            if task_config_path and task_config_path.exists():
                try:
                    with open(task_config_path, 'r') as f:
                        config = json.load(f)
                    print(f"    ✅ Task config loaded successfully")
                    task_loaded = True
                except Exception as e:
                    print(f"    ❌ Task config load failed: {e}")
                    task_loaded = False
            else:
                task_loaded = False
            
            # Test name-based loading
            name_config_filename = f"jolt_config_{dataset_name}.json"
            name_config_path = rm2.path_resolver.get_config_path('jolt', name_config_filename)
            
            print(f"    Name filename: {name_config_filename}")
            print(f"    Name path: {name_config_path}")
            print(f"    Name exists: {name_config_path.exists() if name_config_path else False}")
            
            if name_config_path and name_config_path.exists():
                name_loaded = True
            else:
                name_loaded = False
            
            # Simulate the JOLT wrapper logic
            jolt_config = None
            if task_loaded:
                print(f"    🎯 JOLT would use task-based config")
                jolt_config = "task_config"
            elif name_loaded:
                print(f"    🎯 JOLT would use name-based config")
                jolt_config = "name_config"
            else:
                print(f"    💥 JOLT would use NO config (used_jolt_config: false)")
            
            print(f"    used_jolt_config would be: {jolt_config is not None}")
        
        # Step 3: Test if environment persistence is the issue
        print(f"\n=== Step 3: Testing Environment Persistence ===")
        
        # Test in subprocess to see if environment is passed correctly
        test_script = f'''
import os
import sys
sys.path.insert(0, "{Path(__file__).parent}")

from clam.utils.resource_manager import get_resource_manager

print("Subprocess environment:")
for key in ["CLAM_BASE_DIR", "CLAM_DATASETS_DIR", "CLAM_CACHE_DIR"]:
    print(f"  {{key}}: {{os.environ.get(key, 'NOT_SET')}}")

rm = get_resource_manager()
print(f"Subprocess resource manager base dir: {{rm.path_resolver.get_base_dir()}}")
print(f"Subprocess resource manager configs dir: {{rm.path_resolver.get_configs_dir()}}")

# Test config loading
config_path = rm.path_resolver.get_config_path('jolt', 'jolt_config_task_23.json')
print(f"Subprocess config path: {{config_path}}")
print(f"Subprocess config exists: {{config_path.exists() if config_path else False}}")
'''
        
        result = subprocess.run([sys.executable, '-c', test_script], 
                              capture_output=True, text=True, 
                              env=os.environ.copy())
        
        print("Subprocess output:")
        print(result.stdout)
        if result.stderr:
            print("Subprocess errors:")
            print(result.stderr)
        
        # Step 4: Check file system directly
        print(f"\n=== Step 4: Direct File System Check ===")
        print(f"Direct check of config directory: {clam_configs}")
        print(f"Directory exists: {clam_configs.exists()}")
        
        if clam_configs.exists():
            all_files = list(clam_configs.glob("*"))
            print(f"Files in directory: {len(all_files)}")
            for f in all_files:
                print(f"  - {f.name}")
        
        return True
        
    finally:
        # Restore environment
        for key, original_value in original_env.items():
            if original_value is not None:
                os.environ[key] = original_value
            elif key in os.environ:
                del os.environ[key]
        
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up {temp_dir}: {e}")

if __name__ == "__main__":
    test_exact_remote_server_scenario()