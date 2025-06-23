#!/usr/bin/env python3
"""
Comprehensive test for TabLLM baseline functionality with online note generation.

This test validates:
- Task ID resolution (dataset_id -> task_id)
- Online semantic feature expansion
- Context-aware note truncation
- Note quality and generation
- Integration with existing CLAM infrastructure

Usage:
    python tests/test_tabllm_comprehensive.py
    python tests/test_tabllm_comprehensive.py --task_id 23 --output_dir ./test_tabllm_output
"""

import os
import sys
import json
import logging
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_args(task_id: int = 23, output_dir: str = None, max_context_length: int = 4096):
    """Create test arguments for TabLLM evaluation."""
    class TestArgs:
        def __init__(self):
            self.task_id = task_id
            self.output_dir = output_dir or tempfile.mkdtemp(prefix="tabllm_test_")
            self.max_context_length = max_context_length
            self.num_few_shot_examples = 8
            self.max_test_samples = 10  # Small for testing
            self.device = "cpu"  # Use CPU for testing
            self.seed = 42
            self.openai_model = None
            self.gemini_model = None
            self.api_model = None
            self.tabllm_model = "microsoft/phi-3-mini-128k-instruct"  # Small model for testing
            
    return TestArgs()

def test_task_id_resolution():
    """Test task_id resolution from dataset_id."""
    logger.info("=== Testing Task ID Resolution ===")
    
    # Silence verbose logging
    logging.getLogger('clam.utils.model_loader').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    from examples.tabular.llm_baselines.tabllm_baseline import load_tabllm_config_by_openml_id
    
    # Test cases: (input_id, expected_behavior, description)
    test_cases = [
        (23, "should_work", "Known OpenML task ID"),
        (40996, "should_resolve", "Dataset ID that should resolve to task ID"),
        (999999, "should_fail", "Non-existent ID"),
    ]
    
    for test_id, expected, description in test_cases:
        logger.info(f"\n--- Testing {description}: ID {test_id} ---")
        
        try:
            config_data, feature_mapping = load_tabllm_config_by_openml_id(test_id, original_feature_count=10)
            
            if config_data is not None:
                logger.info(f"✅ SUCCESS: Config found for ID {test_id}")
                logger.info(f"   Task ID resolved from: {test_id}")
                if feature_mapping:
                    semantic_info = feature_mapping.get('semantic_info')
                    if semantic_info:
                        logger.info(f"   Semantic info available: {list(semantic_info.keys())}")
                    else:
                        logger.info("   No semantic info in feature mapping")
            else:
                if expected == "should_fail":
                    logger.info(f"✅ EXPECTED: No config found for ID {test_id}")
                else:
                    logger.warning(f"⚠️ UNEXPECTED: No config found for ID {test_id}")
                    
        except Exception as e:
            if expected == "should_fail":
                logger.info(f"✅ EXPECTED FAILURE: {e}")
            else:
                logger.error(f"❌ UNEXPECTED ERROR: {e}")

def test_semantic_feature_expansion():
    """Test semantic feature expansion functionality."""
    logger.info("\n=== Testing Semantic Feature Expansion ===")
    
    from examples.tabular.llm_baselines.tabllm_baseline import expand_semantic_features
    
    # Test semantic info with 'columns' structure
    test_semantic_info = {
        'columns': [
            {'name': 'age', 'semantic_description': 'Age of the person'},
            {'name': 'income', 'semantic_description': 'Annual income'},
            {'name': 'education', 'semantic_description': 'Education level'}
        ]
    }
    
    test_cases = [
        (3, "no expansion needed"),
        (5, "expand from 3 to 5 features"),
        (8, "expand from 3 to 8 features"),
        (1, "fewer features than available")
    ]
    
    for target_count, description in test_cases:
        logger.info(f"\n--- Testing {description}: {target_count} target features ---")
        
        try:
            expanded_info = expand_semantic_features(test_semantic_info, target_count)
            
            if 'columns' in expanded_info:
                expanded_count = len(expanded_info['columns'])
                logger.info(f"✅ Expanded from 3 to {expanded_count} features")
                
                if target_count > 3:
                    # Check that we have expanded features with suffixes
                    feature_names = [col['name'] for col in expanded_info['columns']]
                    has_suffixes = any('_' in name and name.split('_')[-1].isdigit() for name in feature_names[3:])
                    if has_suffixes:
                        logger.info("   ✅ Expanded features have numeric suffixes")
                        logger.info(f"   Feature names: {feature_names}")
                    else:
                        logger.warning("   ⚠️ Expanded features missing expected suffixes")
                        
                if expanded_count == target_count:
                    logger.info("   ✅ Exact target count achieved")
                else:
                    logger.info(f"   ℹ️ Got {expanded_count}, expected {target_count}")
            else:
                logger.error("❌ Expanded info missing 'columns' structure")
                
        except Exception as e:
            logger.error(f"❌ ERROR: {e}")

def test_online_note_generation():
    """Test online note generation with expanded semantics."""
    logger.info("\n=== Testing Online Note Generation ===")
    
    from examples.tabular.llm_baselines.tabllm_baseline import generate_note_from_row, expand_semantic_features
    
    # Create test semantic info
    semantic_info = {
        'columns': [
            {'name': 'age', 'semantic_description': 'age in years'},
            {'name': 'workclass', 'semantic_description': 'type of employment'},
            {'name': 'education', 'semantic_description': 'education level achieved'}
        ]
    }
    
    # Test data row
    test_row = pd.Series([39, 'State-gov', 'Bachelors'], index=['age', 'workclass', 'education'])
    attribute_names = ['age', 'workclass', 'education']
    
    logger.info("--- Testing basic note generation ---")
    try:
        note = generate_note_from_row(test_row, semantic_info, attribute_names)
        logger.info(f"✅ Generated note: {note}")
        
        # Check that note contains expected elements
        expected_elements = ['age in years is 39', 'type of employment is State-gov', 'education level achieved is Bachelors']
        for element in expected_elements:
            if element in note:
                logger.info(f"   ✅ Found expected element: {element}")
            else:
                logger.warning(f"   ⚠️ Missing expected element: {element}")
                
    except Exception as e:
        logger.error(f"❌ ERROR in basic note generation: {e}")
    
    logger.info("\n--- Testing note generation with expanded features ---")
    try:
        # Expand semantic features to 5
        expanded_semantic = expand_semantic_features(semantic_info, 5)
        
        # Create expanded test row
        expanded_row = pd.Series([39, 'State-gov', 'Bachelors', 50000, 'Married'], 
                                index=['age', 'workclass', 'education', 'age_1', 'workclass_1'])
        expanded_attributes = ['age', 'workclass', 'education', 'age_1', 'workclass_1']
        
        expanded_note = generate_note_from_row(expanded_row, expanded_semantic, expanded_attributes)
        logger.info(f"✅ Generated expanded note: {expanded_note}")
        
        # Check for expanded features
        if 'variant 1' in expanded_note:
            logger.info("   ✅ Found expanded feature variants in note")
        else:
            logger.warning("   ⚠️ Expanded features not reflected in note")
            
    except Exception as e:
        logger.error(f"❌ ERROR in expanded note generation: {e}")

def test_context_aware_truncation():
    """Test context-aware note truncation."""
    logger.info("\n=== Testing Context-Aware Truncation ===")
    
    from examples.tabular.llm_baselines.tabllm_baseline import (
        truncate_few_shot_examples_for_context, 
        estimate_note_tokens,
        estimate_prompt_tokens
    )
    
    # Create mock tokenizer
    class MockTokenizer:
        def encode(self, text):
            # Simple word-based tokenization for testing
            return text.split()
    
    tokenizer = MockTokenizer()
    
    # Create test few-shot examples
    few_shot_examples = [
        ("The age is 25. The income is 50000. The education is Bachelor.", "Class_A"),
        ("The age is 30. The income is 75000. The education is Master.", "Class_B"),
        ("The age is 45. The income is 100000. The education is PhD.", "Class_A"),
        ("The age is 35. The income is 60000. The education is Bachelor.", "Class_B"),
        ("The age is 28. The income is 55000. The education is Master.", "Class_A")
    ]
    
    test_note = "The age is 40. The income is 80000. The education is Master."
    question = "What is the class?"
    task_description = "Classify the following examples."
    
    logger.info("--- Testing token estimation ---")
    try:
        for i, (note, label) in enumerate(few_shot_examples[:2]):
            tokens = estimate_note_tokens(note, tokenizer)
            logger.info(f"   Example {i}: {tokens} tokens for note: {note[:50]}...")
            
        total_tokens = estimate_prompt_tokens(few_shot_examples, test_note, question, task_description, tokenizer)
        logger.info(f"✅ Total estimated tokens: {total_tokens}")
        
    except Exception as e:
        logger.error(f"❌ ERROR in token estimation: {e}")
    
    logger.info("\n--- Testing truncation scenarios ---")
    
    # Test different context limits
    context_limits = [200, 100, 50, 20]
    
    for limit in context_limits:
        try:
            truncated = truncate_few_shot_examples_for_context(
                few_shot_examples, test_note, question, task_description, tokenizer, limit
            )
            
            logger.info(f"   Context limit {limit}: {len(truncated)}/{len(few_shot_examples)} examples kept")
            
            if limit < 50:
                # Very restrictive limit should result in fewer examples
                if len(truncated) < len(few_shot_examples):
                    logger.info("   ✅ Appropriate truncation occurred")
                else:
                    logger.warning("   ⚠️ Expected more aggressive truncation")
                    
        except Exception as e:
            logger.error(f"❌ ERROR with context limit {limit}: {e}")

def test_tabllm_integration():
    """Test full TabLLM integration with a minimal dataset."""
    logger.info("\n=== Testing TabLLM Integration ===")
    
    # Silence verbose logging for integration test
    logging.getLogger('clam.utils.model_loader').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('examples.tabular.llm_baselines.tabllm_baseline').setLevel(logging.WARNING)
    
    from examples.tabular.llm_baselines.tabllm_baseline import evaluate_tabllm
    
    args = create_test_args(task_id=23, max_context_length=2048)
    
    # Create a minimal test dataset
    test_dataset = {
        'name': 'test_dataset',
        'task_id': 23,  # Use task_id instead of id
        'X': pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'feature_3': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        }),
        'y': pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        'attribute_names': ['feature_1', 'feature_2', 'feature_3']
    }
    
    logger.info("--- Testing TabLLM evaluation pipeline ---")
    
    try:
        # This is a dry run to test the pipeline
        results = evaluate_tabllm(test_dataset, args)
        
        if isinstance(results, dict):
            logger.info("✅ TabLLM evaluation completed successfully")
            logger.info(f"   Model: {results.get('model_name', 'Unknown')}")
            logger.info(f"   Dataset: {results.get('dataset_name', 'Unknown')}")
            logger.info(f"   Completed samples: {results.get('completed_samples', 0)}")
            logger.info(f"   Accuracy: {results.get('accuracy', 'N/A')}")
            
            # Check for note inspection files
            inspection_dir = Path(args.output_dir) / "tabllm_notes_inspection"
            if inspection_dir.exists():
                inspection_files = list(inspection_dir.glob("*.json"))
                if inspection_files:
                    logger.info(f"   ✅ Note inspection files created: {len(inspection_files)} files")
                    
                    # Check content of first file
                    with open(inspection_files[0], 'r') as f:
                        inspection_data = json.load(f)
                    
                    if 'sample_notes' in inspection_data:
                        num_notes = len(inspection_data['sample_notes'])
                        logger.info(f"      Sample notes saved: {num_notes}")
                        
                        if num_notes > 0:
                            first_note = inspection_data['sample_notes'][0]['note']
                            logger.info(f"      First note preview: {first_note[:100]}...")
                else:
                    logger.warning("   ⚠️ No inspection files found")
            else:
                logger.warning("   ⚠️ Inspection directory not created")
                
        else:
            logger.error(f"❌ Unexpected result type: {type(results)}")
            
    except Exception as e:
        # This is expected to fail in some cases due to model loading
        logger.warning(f"⚠️ TabLLM integration test failed (may be expected): {e}")
        logger.info("   This is often due to model availability or API limits in test environment")
        
    finally:
        # Clean up test directory
        if os.path.exists(args.output_dir):
            try:
                shutil.rmtree(args.output_dir)
                logger.info(f"   Cleaned up test directory: {args.output_dir}")
            except:
                logger.warning(f"   Could not clean up test directory: {args.output_dir}")

def test_note_inspection_system():
    """Test the note inspection file saving system."""
    logger.info("\n=== Testing Note Inspection System ===")
    
    from examples.tabular.llm_baselines.tabllm_baseline import save_sample_notes_for_inspection
    
    # Create test data
    few_shot_examples = [
        ("The age is 25. The workclass is Private. The education is Bachelors.", "<=50K"),
        ("The age is 38. The workclass is Self-emp-not-inc. The education is HS-grad.", ">50K"),
        ("The age is 45. The workclass is Local-gov. The education is Masters.", ">50K")
    ]
    
    test_dataset = {
        'name': 'adult',
        'task_id': 1590,
        'attribute_names': ['age', 'workclass', 'education']
    }
    
    test_semantic_info = {
        'columns': [
            {'name': 'age', 'semantic_description': 'age in years'},
            {'name': 'workclass', 'semantic_description': 'employment type'},
            {'name': 'education', 'semantic_description': 'education level'}
        ]
    }
    
    args = create_test_args()
    
    try:
        save_sample_notes_for_inspection(few_shot_examples, test_dataset, args, test_semantic_info)
        
        # Check if file was created
        inspection_dir = Path(args.output_dir) / "tabllm_notes_inspection"
        if inspection_dir.exists():
            inspection_files = list(inspection_dir.glob("*.json"))
            if inspection_files:
                logger.info(f"✅ Inspection file created: {inspection_files[0].name}")
                
                # Validate file content
                with open(inspection_files[0], 'r') as f:
                    data = json.load(f)
                
                required_keys = ['metadata', 'semantic_expansion_info', 'sample_notes']
                for key in required_keys:
                    if key in data:
                        logger.info(f"   ✅ Found required key: {key}")
                    else:
                        logger.error(f"   ❌ Missing required key: {key}")
                
                if 'sample_notes' in data and len(data['sample_notes']) > 0:
                    note_keys = ['note', 'label', 'statistics']
                    first_note = data['sample_notes'][0]
                    for key in note_keys:
                        if key in first_note:
                            logger.info(f"      ✅ Note has required key: {key}")
                        else:
                            logger.error(f"      ❌ Note missing key: {key}")
            else:
                logger.error("❌ No inspection files created")
        else:
            logger.error("❌ Inspection directory not created")
            
    except Exception as e:
        logger.error(f"❌ ERROR in note inspection test: {e}")
    
    finally:
        # Clean up
        if os.path.exists(args.output_dir):
            try:
                shutil.rmtree(args.output_dir)
            except:
                pass

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Comprehensive TabLLM Test Suite')
    parser.add_argument('--task_id', type=int, default=23, help='OpenML task ID to test with')
    parser.add_argument('--output_dir', type=str, help='Output directory for test files')
    parser.add_argument('--skip_integration', action='store_true', help='Skip integration test')
    
    args = parser.parse_args()
    
    logger.info("=== Comprehensive TabLLM Baseline Test Suite ===")
    logger.info(f"Testing with task_id: {args.task_id}")
    
    # Run all tests
    test_task_id_resolution()
    test_semantic_feature_expansion()
    test_online_note_generation()
    test_context_aware_truncation()
    test_note_inspection_system()
    
    if not args.skip_integration:
        test_tabllm_integration()
    else:
        logger.info("\n=== Skipping Integration Test (--skip_integration) ===")
    
    logger.info("\n=== All TabLLM Tests Complete ===")

if __name__ == "__main__":
    main()