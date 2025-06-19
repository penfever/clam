#!/usr/bin/env python3
"""
Test script to verify VLM response parsing with problematic strings.
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clam.utils.vlm_prompting import parse_vlm_response
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Test cases from the bio script logs
test_cases = [
    'Class: humpback_whale | Reasoning',
    'Class: blue_whale | Reasoning: The',
    'Class: humpback_whale | Reasoning',
    'Class: blue_whale | Reasoning: The',
    'Class: blue_whale | Reasoning: The',
    'Class: chimp | Reasoning: The red',
    'Class: spider_monkey | Reasoning: The',
    'Class: persian_cat | Reasoning: The'
]

# Mock class names for biological classification
bio_class_names = [
    'antelope', 'bear', 'blue_whale', 'cat', 'chimp', 
    'humpback_whale', 'persian_cat', 'spider_monkey'
]

def test_parsing():
    """Test the VLM response parsing with problematic strings."""
    print("Testing VLM response parsing...")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: '{test_case}'")
        try:
            # Test with semantic names (use_semantic_names=True)
            result = parse_vlm_response(
                test_case, 
                unique_classes=bio_class_names, 
                use_semantic_names=True
            )
            print(f"  Parsed result: {result}")
            
            # Check if the result is expected
            expected_class = None
            if 'humpback_whale' in test_case:
                expected_class = 'humpback_whale'
            elif 'blue_whale' in test_case:
                expected_class = 'blue_whale'
            elif 'chimp' in test_case:
                expected_class = 'chimp'
            elif 'spider_monkey' in test_case:
                expected_class = 'spider_monkey'
            elif 'persian_cat' in test_case:
                expected_class = 'persian_cat'
            
            if expected_class and result == expected_class:
                print(f"  ✅ SUCCESS: Correctly parsed '{expected_class}'")
            elif expected_class:
                print(f"  ❌ FAILURE: Expected '{expected_class}', got '{result}'")
            else:
                print(f"  ⚠️  WARNING: Could not determine expected class")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed.")

if __name__ == "__main__":
    test_parsing()