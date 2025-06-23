#!/usr/bin/env python3
"""
Test JOLT evaluation with debug output to see config loading in action.
"""

import sys
sys.path.insert(0, '.')
from examples.tabular.llm_baselines.jolt.official_jolt_wrapper import evaluate_jolt_official
from argparse import Namespace

# Create minimal test args
args = Namespace(
    device='cpu',
    seed=42,
    jolt_model='microsoft/DialoGPT-small',  # Small model for quick test
    gpu_index=0
)

# Create test dataset
test_dataset = {
    'id': '23',
    'name': 'cmc',
    'data_source': 'openml'
}

print('Testing JOLT config loading in actual evaluation...')
try:
    # This will show the debug output from config loading
    result = evaluate_jolt_official(test_dataset, args)
    print(f'JOLT evaluation result keys: {list(result.keys())}')
    print(f'used_jolt_config: {result.get("used_jolt_config", "KEY_NOT_FOUND")}')
except Exception as e:
    print(f'Error during JOLT evaluation: {e}')
    import traceback
    traceback.print_exc()