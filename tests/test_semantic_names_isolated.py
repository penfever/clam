#!/usr/bin/env python
"""
Isolated test for semantic names functionality in CLAM.

This test isolates and debugs the semantic names feature to ensure it works correctly.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clam.models.clam_tsne import ClamTsneClassifier
from clam.utils.vlm_prompting import create_classification_prompt, parse_vlm_response
from clam.utils.class_name_utils import extract_class_names_from_labels

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_semantic_class_names(task_id: int, num_classes: int):
    """Load semantic class names from semantic data directory using general search."""
    # Try to find semantic file using the general search
    try:
        from clam.utils.metadata_loader import get_metadata_loader
        loader = get_metadata_loader()
        semantic_file = loader.detect_metadata_file(task_id)
    except Exception as e:
        logger.debug(f"Could not use metadata loader: {e}")
        # Fallback to hardcoded path
        semantic_file = Path(__file__).parent.parent / "data" / "cc18_semantic" / f"{task_id}.json"
    
    if not semantic_file or not semantic_file.exists():
        logger.info(f"No semantic file found for task {task_id}")
        return None
    
    try:
        with open(semantic_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Try different structures to extract class names
        class_names = None
        
        # Method 1: target_values (maps numeric labels to semantic names)
        if 'target_values' in data:
            target_values = data['target_values']
            if isinstance(target_values, dict):
                # Sort by numeric key to get correct order
                sorted_items = sorted(target_values.items(), key=lambda x: int(x[0]))
                class_names = [item[1] for item in sorted_items]
                logger.info(f"Method 1 (target_values): Found class names: {class_names}")
        
        # Method 2: target_classes (list with name/meaning)
        if class_names is None and 'target_classes' in data:
            target_classes = data['target_classes']
            if isinstance(target_classes, list):
                # Use 'meaning' if available, otherwise 'name'
                class_names = []
                for tc in target_classes:
                    if isinstance(tc, dict):
                        name = tc.get('meaning', tc.get('name', ''))
                        class_names.append(name)
                    else:
                        class_names.append(str(tc))
                logger.info(f"Method 2 (target_classes): Found class names: {class_names}")
        
        # Method 3: instances_per_class keys
        if class_names is None and 'instances_per_class' in data:
            instances_per_class = data['instances_per_class']
            if isinstance(instances_per_class, dict):
                class_names = list(instances_per_class.keys())
                logger.info(f"Method 3 (instances_per_class): Found class names: {class_names}")
        
        # Validate and truncate to match number of classes
        if class_names:
            # Clean up class names (remove extra whitespace, etc.)
            class_names = [name.strip() for name in class_names if name.strip()]
            
            # Truncate to match actual number of classes
            if len(class_names) > num_classes:
                logger.warning(f"Found {len(class_names)} class names but dataset has {num_classes} classes. Truncating.")
                class_names = class_names[:num_classes]
            elif len(class_names) < num_classes:
                logger.warning(f"Found {len(class_names)} class names but dataset has {num_classes} classes. May be missing some.")
            
            logger.info(f"Final semantic class names: {class_names}")
            return class_names
        
        logger.warning(f"Could not extract class names from semantic file for task {task_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error loading semantic file for task {task_id}: {e}")
        return None


def test_semantic_names_loading():
    """Test semantic names loading for task 23."""
    print("=== Testing Semantic Names Loading ===")
    
    # Load dataset
    data = fetch_openml(data_id=23, as_frame=True, parser='auto')
    X, y = data.data, data.target
    
    # Convert target to numeric
    if hasattr(y, 'cat'):
        y_numeric = y.cat.codes
    else:
        unique_classes = sorted(y.unique())
        y_numeric = y.map({cls: i for i, cls in enumerate(unique_classes)})
    
    print(f"Dataset shape: {X.shape}")
    print(f"Original target classes: {sorted(y.unique())}")
    print(f"Numeric target range: {y_numeric.min()} to {y_numeric.max()}")
    print(f"Number of classes: {len(np.unique(y_numeric))}")
    
    # Load semantic class names
    semantic_names = load_semantic_class_names(23, len(np.unique(y_numeric)))
    if semantic_names:
        print(f"✅ Loaded semantic names: {semantic_names}")
    else:
        print("❌ Failed to load semantic names")
        return False
    
    return True


def test_clam_with_semantic_names():
    """Test CLAM classifier with semantic names."""
    print("\n=== Testing CLAM with Semantic Names ===")
    
    # Load dataset
    data = fetch_openml(data_id=23, as_frame=True, parser='auto')
    X, y = data.data, data.target
    
    # Convert target to numeric
    if hasattr(y, 'cat'):
        y_numeric = y.cat.codes
    else:
        unique_classes = sorted(y.unique())
        y_numeric = y.map({cls: i for i, cls in enumerate(unique_classes)})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_numeric, test_size=0.1, random_state=42, stratify=y_numeric
    )
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Load semantic class names
    semantic_names = load_semantic_class_names(23, len(np.unique(y_numeric)))
    if not semantic_names:
        print("❌ Cannot test without semantic names")
        return False
    
    # Create classifier
    print(f"Creating classifier with semantic names: {semantic_names}")
    classifier = ClamTsneClassifier(
        modality='tabular',
        vlm_model_id='Qwen/Qwen2.5-VL-3B-Instruct',
        use_semantic_names=True,
        tsne_perplexity=15,
        tsne_n_iter=300,
        max_vlm_image_size=1024,
        seed=42
    )
    
    # Fit with semantic class names
    print("Fitting classifier with semantic class names...")
    try:
        classifier.fit(X_train, y_train, X_test, class_names=semantic_names)
        print("✅ Successfully fitted classifier")
    except Exception as e:
        print(f"❌ Error fitting classifier: {e}")
        return False
    
    # Check if semantic names were stored
    if hasattr(classifier, 'class_to_semantic'):
        print(f"Class to semantic mapping: {classifier.class_to_semantic}")
    else:
        print("❌ No class_to_semantic mapping found")
    
    # Test prediction on a few samples
    print("Testing predictions on 3 samples...")
    try:
        predictions = classifier.predict(X_test[:3], y_test[:3], return_detailed=True)
        
        if isinstance(predictions, dict) and 'predictions' in predictions:
            pred_values = predictions['predictions']
            print(f"✅ Predictions: {pred_values}")
            
            # Check if responses use semantic names
            if 'responses' in predictions:
                responses = predictions['responses']
                print(f"Sample responses:")
                for i, response in enumerate(responses[:3]):
                    print(f"  Response {i}: {response[:100]}...")
                    # Check if any semantic names appear in the response
                    has_semantic = any(name.lower() in response.lower() for name in semantic_names)
                    print(f"    Contains semantic names: {has_semantic}")
            
        else:
            print(f"❌ Unexpected predictions format: {type(predictions)}")
            return False
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return False
    
    return True


def test_vlm_prompting_with_semantic_names():
    """Test VLM prompting directly with semantic names."""
    print("\n=== Testing VLM Prompting with Semantic Names ===")
    
    # Load semantic names
    semantic_names = load_semantic_class_names(23, 3)
    if not semantic_names:
        print("❌ Cannot test without semantic names")
        return False
    
    # Test prompt creation
    print("Testing prompt creation with semantic names...")
    prompt_semantic = create_classification_prompt(
        class_names=semantic_names,
        modality='tabular',
        dataset_description="Contraceptive method choice dataset",
        use_semantic_names=True
    )
    
    prompt_standard = create_classification_prompt(
        class_names=semantic_names,
        modality='tabular',
        dataset_description="Contraceptive method choice dataset",
        use_semantic_names=False
    )
    
    print("Semantic prompt snippet:")
    print(prompt_semantic[:300] + "...")
    print("\nStandard prompt snippet:")
    print(prompt_standard[:300] + "...")
    
    # Check if semantic names appear in semantic prompt
    has_semantic_in_prompt = any(name in prompt_semantic for name in semantic_names)
    print(f"✅ Semantic names appear in semantic prompt: {has_semantic_in_prompt}")
    
    # Test response parsing
    print("\nTesting response parsing...")
    test_responses = [
        f'Class: "{semantic_names[0]}" | Reasoning: Test response',
        f'The answer is {semantic_names[1]}',
        f'{semantic_names[2]} because of the pattern'
    ]
    
    for i, response in enumerate(test_responses):
        parsed = parse_vlm_response(
            response, 
            unique_classes=semantic_names, 
            use_semantic_names=True, 
            task_type='classification'
        )
        expected = semantic_names[i]
        print(f"Response: '{response}' -> Parsed: '{parsed}' (Expected: '{expected}')")
        print(f"  ✅ Match: {parsed == expected}")
    
    return True


def main():
    """Run all semantic names tests."""
    print("🧪 CLAM Semantic Names Isolated Test")
    print("=" * 50)
    
    tests = [
        ("Semantic Names Loading", test_semantic_names_loading),
        ("CLAM with Semantic Names", test_clam_with_semantic_names), 
        ("VLM Prompting with Semantic Names", test_vlm_prompting_with_semantic_names)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n❌ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)