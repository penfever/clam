#!/usr/bin/env python3
"""
Debug script to test JOLT baseline integration with the regression evaluation pipeline.
This mimics what the regression baseline script does.
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

def test_jolt_baseline_evaluation():
    """Test the JOLT baseline evaluation like the regression script does."""
    
    # Silence verbose logging
    logging.getLogger('clam.utils.model_loader').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('vllm').setLevel(logging.WARNING)
    
    # Import the key functions
    from examples.tabular.llm_baselines.jolt_baseline import load_jolt_config_by_openml_id
    
    # Test with the task that was failing: 363373
    test_task_id = 363373
    
    logger.info(f"Testing JOLT baseline for task ID: {test_task_id}")
    
    # Test the config lookup (this is what the baseline does first)
    try:
        # Test with different feature counts to see if there's a validation issue
        for feature_count in [6, 16, None]:  # 6 is from config, 16 is from log, None is no validation
            logger.info(f"Testing with feature count: {feature_count}")
            
            config_data, feature_mapping = load_jolt_config_by_openml_id(
                test_task_id, 
                original_feature_count=feature_count
            )
            
            if config_data is not None:
                logger.info(f"✅ Config found with feature count {feature_count}")
                logger.info(f"Config expects {config_data.get('num_features')} features")
                logger.info(f"Config has {config_data.get('num_classes')} classes")
            else:
                logger.error(f"❌ Config NOT found with feature count {feature_count}")
                
    except Exception as e:
        logger.error(f"❌ Config lookup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

def test_directory_paths():
    """Test the directory paths to ensure they match what the baseline expects."""
    
    logger.info("Testing directory paths...")
    
    # Test the JOLT directory path
    jolt_dir = os.path.join(project_root, "examples", "tabular", "llm_baselines", "jolt")
    logger.info(f"JOLT directory: {jolt_dir}")
    logger.info(f"Directory exists: {os.path.exists(jolt_dir)}")
    
    # Check for task 363373 config
    config_path = os.path.join(jolt_dir, "jolt_config_task_363373.json")
    logger.info(f"Config file path: {config_path}")
    logger.info(f"Config file exists: {os.path.exists(config_path)}")
    
    # Check the working directory assumption
    current_dir = os.path.dirname(os.path.abspath(__file__))
    expected_jolt_dir = os.path.join(current_dir, "examples", "tabular", "llm_baselines", "jolt")
    logger.info(f"Expected JOLT dir (from script location): {expected_jolt_dir}")
    logger.info(f"Expected dir exists: {os.path.exists(expected_jolt_dir)}")
    
def test_imports():
    """Test if all required imports work."""
    
    logger.info("Testing imports...")
    
    try:
        from examples.tabular.llm_baselines.jolt_baseline import load_jolt_config_by_openml_id
        logger.info("✅ Successfully imported load_jolt_config_by_openml_id")
    except ImportError as e:
        logger.error(f"❌ Failed to import load_jolt_config_by_openml_id: {e}")
    
    try:
        from examples.tabular.llm_baselines.jolt_baseline import evaluate_jolt
        logger.info("✅ Successfully imported evaluate_jolt")
    except ImportError as e:
        logger.error(f"❌ Failed to import evaluate_jolt: {e}")

def main():
    """Main test function."""
    logger.info("=== JOLT Baseline Integration Debug Test ===")
    
    logger.info("\n1. Testing imports...")
    test_imports()
    
    logger.info("\n2. Testing directory paths...")
    test_directory_paths()
    
    logger.info("\n3. Testing JOLT baseline evaluation...")
    test_jolt_baseline_evaluation()
    
    logger.info("\n=== Debug test complete ===")

if __name__ == "__main__":
    main()