#!/usr/bin/env python3
"""
Debug script to test JOLT config lookup functionality.
This will help identify why the JOLT baseline can't find configs.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_jolt_config_lookup():
    """Test the JOLT config lookup functionality directly."""
    
    # Import the function from JOLT baseline
    from examples.tabular.llm_baselines.jolt_baseline import load_jolt_config_by_openml_id
    
    # Test with task 363373 that we know has a config
    test_task_id = 363373
    
    logger.info(f"Testing JOLT config lookup for task ID: {test_task_id}")
    
    # Check if the config file exists first
    jolt_dir = os.path.join(project_root, "examples", "tabular", "llm_baselines", "jolt")
    config_path = os.path.join(jolt_dir, f"jolt_config_task_{test_task_id}.json")
    
    logger.info(f"Looking for config file at: {config_path}")
    logger.info(f"File exists: {os.path.exists(config_path)}")
    
    if os.path.exists(config_path):
        # Try to read the file directly
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            logger.info(f"Direct file read successful. Config has {config_data.get('num_features')} features, {config_data.get('num_classes')} classes")
        except Exception as e:
            logger.error(f"Error reading config file directly: {e}")
    
    # Now test the lookup function
    try:
        config_data, feature_mapping = load_jolt_config_by_openml_id(test_task_id, original_feature_count=6)
        
        if config_data is not None:
            logger.info("✅ JOLT config lookup SUCCESSFUL!")
            logger.info(f"Config data keys: {list(config_data.keys())}")
            logger.info(f"Dataset name: {config_data.get('dataset_name')}")
            logger.info(f"Number of features: {config_data.get('num_features')}")
            logger.info(f"Number of classes: {config_data.get('num_classes')}")
            logger.info(f"Task type: {'regression' if config_data.get('num_classes') == 1 else 'classification'}")
        else:
            logger.error("❌ JOLT config lookup FAILED - returned None")
            
    except Exception as e:
        logger.error(f"❌ JOLT config lookup FAILED with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())

def test_directory_structure():
    """Test the directory structure and file listing."""
    
    jolt_dir = os.path.join(project_root, "examples", "tabular", "llm_baselines", "jolt")
    logger.info(f"JOLT directory: {jolt_dir}")
    logger.info(f"Directory exists: {os.path.exists(jolt_dir)}")
    
    if os.path.exists(jolt_dir):
        # List all jolt config files
        config_files = [f for f in os.listdir(jolt_dir) if f.startswith('jolt_config_task_') and f.endswith('.json')]
        logger.info(f"Found {len(config_files)} JOLT config files")
        
        # Show first few and any that match our test case
        for i, config_file in enumerate(sorted(config_files)[:5]):
            logger.info(f"  {i+1}. {config_file}")
        
        # Specifically look for task 363373
        target_file = "jolt_config_task_363373.json"
        if target_file in config_files:
            logger.info(f"✅ Found target config file: {target_file}")
        else:
            logger.error(f"❌ Target config file NOT found: {target_file}")

def test_openml_task_mapping():
    """Test if there's still an old openml_task_mapping.json file that might interfere."""
    
    jolt_dir = os.path.join(project_root, "examples", "tabular", "llm_baselines", "jolt")
    mapping_path = os.path.join(jolt_dir, "openml_task_mapping.json")
    
    logger.info(f"Checking for old mapping file at: {mapping_path}")
    logger.info(f"Old mapping file exists: {os.path.exists(mapping_path)}")
    
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r') as f:
                mapping_data = json.load(f)
            logger.info(f"Old mapping file contains {len(mapping_data)} entries")
            
            # Check if task 363373 is in the old mapping
            found_363373 = False
            for dataset_name, task_id in mapping_data.items():
                if task_id == 363373:
                    logger.info(f"Task 363373 found in old mapping with dataset name: {dataset_name}")
                    found_363373 = True
                    break
            
            if not found_363373:
                logger.info("Task 363373 NOT found in old mapping file")
                
        except Exception as e:
            logger.error(f"Error reading old mapping file: {e}")

def main():
    """Main test function."""
    logger.info("=== JOLT Config Lookup Debug Test ===")
    
    logger.info("\n1. Testing directory structure...")
    test_directory_structure()
    
    logger.info("\n2. Testing old mapping file...")
    test_openml_task_mapping()
    
    logger.info("\n3. Testing JOLT config lookup...")
    test_jolt_config_lookup()
    
    logger.info("\n=== Debug test complete ===")

if __name__ == "__main__":
    main()