#!/usr/bin/env python3
"""
Simple debug script to test JOLT config lookup without all the extra logging.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_jolt_config_lookup():
    """Test the JOLT config lookup functionality directly."""
    
    # Silence the model loader
    logging.getLogger('clam.utils.model_loader').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('vllm').setLevel(logging.WARNING)
    
    # Import the function from JOLT baseline
    from examples.tabular.llm_baselines.jolt_baseline import load_jolt_config_by_openml_id
    
    # Test with task 363373 that we know has a config
    test_task_id = 363373
    
    logger.info(f"Testing JOLT config lookup for task ID: {test_task_id}")
    
    # Check if the config file exists first
    jolt_dir = os.path.join(project_root, "examples", "tabular", "llm_baselines", "jolt")
    config_path = os.path.join(jolt_dir, f"jolt_config_task_{test_task_id}.json")
    
    logger.info(f"Config file exists: {os.path.exists(config_path)}")
    
    # Now test the lookup function
    try:
        config_data, feature_mapping = load_jolt_config_by_openml_id(test_task_id, original_feature_count=6)
        
        if config_data is not None:
            logger.info("✅ JOLT config lookup SUCCESSFUL!")
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

if __name__ == "__main__":
    test_jolt_config_lookup()