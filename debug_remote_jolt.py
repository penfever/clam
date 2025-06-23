#!/usr/bin/env python3
"""
Debug script to check JOLT config loading on remote server.
"""

import sys
import os
from pathlib import Path

# Print environment
print('=== Environment Variables ===')
for var in ['CLAM_BASE_DIR', 'CLAM_CONFIGS_DIR', 'CLAM_CACHE_DIR', 'CLAM_DATASETS_DIR']:
    print(f'{var}: {os.environ.get(var, "NOT_SET")}')

# Test resource manager
sys.path.insert(0, '.')
from clam.utils.resource_manager import get_resource_manager
rm = get_resource_manager()

print('\n=== Resource Manager Paths ===')
print(f'Base dir: {rm.path_resolver.get_base_dir()}')
print(f'Configs dir: {rm.path_resolver.get_configs_dir()}')
print(f'JOLT configs dir: {rm.path_resolver.get_configs_dir() / "jolt"}')

# Check JOLT configs directory
jolt_dir = rm.path_resolver.get_configs_dir() / 'jolt'
print(f'\n=== JOLT Directory Check ===')
print(f'JOLT dir exists: {jolt_dir.exists()}')

if jolt_dir.exists():
    import glob
    all_files = list(jolt_dir.glob('*.json'))
    task_files = [f for f in all_files if f.name.startswith('jolt_config_task_')]
    print(f'Total JSON files: {len(all_files)}')
    print(f'Task-based files: {len(task_files)}')
    if task_files:
        print(f'Sample task files: {[f.name for f in task_files[:5]]}')

# Test config loading for task 23
print(f'\n=== Config Loading Test for Task 23 ===')
task_config_filename = 'jolt_config_task_23.json'
jolt_config_path = rm.path_resolver.get_config_path('jolt', task_config_filename)

print(f'Looking for: {task_config_filename}')
print(f'Path returned: {jolt_config_path}')
print(f'File exists: {jolt_config_path.exists() if jolt_config_path else False}')

if jolt_config_path and jolt_config_path.exists():
    import json
    with open(jolt_config_path, 'r') as f:
        config = json.load(f)
    print(f'✅ Config loaded successfully')
    print(f'Task prefix: {config.get("task_prefix", "No prefix")[:100]}...')
else:
    print(f'❌ Config not found')