#!/usr/bin/env python3
"""
Comprehensive test for JOLT baseline functionality after the fix.
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

def test_feature_count_scenarios():
    """Test various feature count scenarios."""
    
    # Silence verbose logging
    logging.getLogger('clam.utils.model_loader').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    from examples.tabular.llm_baselines.jolt_baseline import load_jolt_config_by_openml_id
    
    test_task_id = 363373
    
    logger.info("Testing various feature count scenarios...")
    
    test_cases = [
        (6, "exact match"),
        (16, "expanded features (text preprocessing)"),
        (20, "many expanded features"),
        (3, "fewer features (should fail)"),
        (None, "no validation")
    ]
    
    for feature_count, description in test_cases:
        logger.info(f"\n--- Testing {description}: {feature_count} features ---")
        
        try:
            config_data, feature_mapping = load_jolt_config_by_openml_id(
                test_task_id, 
                original_feature_count=feature_count
            )
            
            if config_data is not None:
                logger.info(f"✅ SUCCESS: Config found")
                logger.info(f"   Expected features: {config_data.get('num_features')}")
                logger.info(f"   Actual features: {feature_count}")
                logger.info(f"   Classes: {config_data.get('num_classes')}")
                logger.info(f"   Task type: {'regression' if config_data.get('num_classes') == 1 else 'classification'}")
            else:
                logger.error(f"❌ FAILED: Config not found")
                
        except ValueError as e:
            if feature_count == 3:
                logger.info(f"✅ EXPECTED FAILURE: {e}")
            else:
                logger.error(f"❌ UNEXPECTED FAILURE: {e}")
        except Exception as e:
            logger.error(f"❌ UNEXPECTED ERROR: {e}")

def test_regression_tasks():
    """Test multiple regression task configs."""
    
    logging.getLogger('clam.utils.model_loader').setLevel(logging.WARNING)
    
    from examples.tabular.llm_baselines.jolt_baseline import load_jolt_config_by_openml_id
    
    # Check some regression task IDs from the logs
    regression_task_ids = [361085, 361086, 361087, 361088, 363373, 363374]
    
    logger.info("\nTesting multiple regression tasks...")
    
    for task_id in regression_task_ids:
        logger.info(f"\n--- Testing task {task_id} ---")
        
        try:
            config_data, feature_mapping = load_jolt_config_by_openml_id(
                task_id, 
                original_feature_count=None  # No validation for this test
            )
            
            if config_data is not None:
                logger.info(f"✅ Task {task_id}: Config found")
                logger.info(f"   Dataset: {config_data.get('dataset_name')}")
                logger.info(f"   Features: {config_data.get('num_features')}")
                logger.info(f"   Classes: {config_data.get('num_classes')}")
                logger.info(f"   Type: {'regression' if config_data.get('num_classes') == 1 else 'classification'}")
                
                # Check if it's correctly identified as regression
                if config_data.get('num_classes') == 1:
                    logger.info("   ✅ Correctly identified as regression")
                else:
                    logger.warning("   ⚠️ Not identified as regression")
            else:
                logger.error(f"❌ Task {task_id}: Config not found")
                
        except Exception as e:
            logger.error(f"❌ Task {task_id}: Error - {e}")

def main():
    """Main test function."""
    logger.info("=== Comprehensive JOLT Baseline Test ===")
    
    test_feature_count_scenarios()
    test_regression_tasks()
    
    logger.info("\n=== Test complete ===")

if __name__ == "__main__":
    main()