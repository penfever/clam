#!/usr/bin/env python3
"""
Test just the JOLT config loading part without running full evaluation.
"""

import sys
import logging
sys.path.insert(0, '.')

# Set up logging to see the debug output
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Test dataset info (just metadata, no actual data)
test_dataset = {
    'id': '23',
    'name': 'cmc',
    'data_source': 'openml'
}

print('Testing JOLT config loading logic directly...')

# This is the exact config loading code from the JOLT wrapper
jolt_config = None
try:
    from clam.utils.resource_manager import get_resource_manager
    rm = get_resource_manager()
    
    # Debug: Print path information (from my added debug code)
    logger.info(f"JOLT config loading debug:")
    logger.info(f"  Base dir: {rm.path_resolver.get_base_dir()}")
    logger.info(f"  Configs dir: {rm.path_resolver.get_configs_dir()}")
    logger.info(f"  JOLT configs dir: {rm.path_resolver.get_configs_dir() / 'jolt'}")
    
    # First try the task-based naming convention (newer format)
    task_config_filename = f"jolt_config_task_{test_dataset['id']}.json"
    jolt_config_path = rm.path_resolver.get_config_path('jolt', task_config_filename)
    
    logger.info(f"  Looking for task config: {task_config_filename}")
    logger.info(f"  Task config path: {jolt_config_path}")
    logger.info(f"  Task config exists: {jolt_config_path.exists() if jolt_config_path else False}")
    
    if jolt_config_path and jolt_config_path.exists():
        import json
        with open(jolt_config_path, 'r') as f:
            jolt_config = json.load(f)
        logger.info(f"✅ Using JOLT metadata for task {test_dataset['id']} ({test_dataset['name']}) from managed config")
    else:
        # Fallback to dataset name-based naming (legacy format)
        name_config_filename = f"jolt_config_{test_dataset['name']}.json"
        jolt_config_path = rm.path_resolver.get_config_path('jolt', name_config_filename)
        
        logger.info(f"  Looking for name config: {name_config_filename}")
        logger.info(f"  Name config path: {jolt_config_path}")
        logger.info(f"  Name config exists: {jolt_config_path.exists() if jolt_config_path else False}")
        
        if jolt_config_path and jolt_config_path.exists():
            import json
            with open(jolt_config_path, 'r') as f:
                jolt_config = json.load(f)
            logger.info(f"✅ Using JOLT metadata for {test_dataset['name']} from managed config (legacy naming)")
        else:
            logger.info(f"❌ No JOLT metadata found for task {test_dataset['id']} ({test_dataset['name']}), using default approach")
except Exception as e:
    logger.error(f"Could not load JOLT config for task {test_dataset['id']} ({test_dataset['name']}): {e}")
    import traceback
    traceback.print_exc()

# Print the final result
print(f"\n=== FINAL RESULT ===")
print(f"jolt_config loaded: {jolt_config is not None}")
print(f"used_jolt_config would be: {jolt_config is not None}")

if jolt_config:
    print(f"Config task_prefix: {jolt_config.get('task_prefix', 'No prefix')[:100]}...")
    print(f"Config keys: {list(jolt_config.keys())}")
else:
    print("No config loaded - this explains why used_jolt_config is false")