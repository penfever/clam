"""
Metadata validation utilities for TabLLM and JOLT baselines.

This module provides functions to validate that required metadata files exist
and contain valid information for running TabLLM and JOLT baselines.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


def get_project_root() -> str:
    """Get the project root directory."""
    # Find project root by looking for key directories
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    
    # Go up directories until we find the clam package root
    while current_dir != "/":
        if os.path.exists(os.path.join(current_dir, "clam")) and os.path.exists(os.path.join(current_dir, "data")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    # Fallback: use relative path from utils directory
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def validate_tabllm_metadata(openml_task_id: int, feature_count: Optional[int] = None) -> Dict[str, Any]:
    """
    Validate TabLLM metadata for a given OpenML task.
    
    Args:
        openml_task_id: OpenML task ID to validate
        feature_count: Expected number of features (optional)
        
    Returns:
        Dictionary with validation results:
        - 'valid': bool - whether metadata is valid
        - 'missing_files': list - list of missing required files
        - 'errors': list - list of validation errors
        - 'warnings': list - list of warnings
        - 'dataset_name': str - mapped dataset name if found
    """
    project_root = get_project_root()
    tabllm_dir = os.path.join(project_root, "examples", "tabular", "llm_baselines", "tabllm_like")
    semantic_dir = os.path.join(project_root, "data", "cc18_semantic")
    
    result = {
        'valid': True,
        'missing_files': [],
        'errors': [],
        'warnings': [],
        'dataset_name': None
    }
    
    # Check OpenML task mapping
    mapping_path = os.path.join(tabllm_dir, "openml_task_mapping.json")
    if not os.path.exists(mapping_path):
        result['valid'] = False
        result['missing_files'].append(mapping_path)
        result['errors'].append("OpenML task mapping file not found")
        return result
    
    try:
        with open(mapping_path, 'r') as f:
            task_mapping = json.load(f)
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Error loading OpenML task mapping: {e}")
        return result
    
    # Find dataset name for this task ID
    dataset_name = None
    for name, task_id in task_mapping.items():
        if task_id == openml_task_id:
            dataset_name = name
            break
    
    if dataset_name is None:
        result['valid'] = False
        result['errors'].append(f"No TabLLM config found for OpenML task ID {openml_task_id}")
        return result
    
    result['dataset_name'] = dataset_name
    
    # Check template file
    template_path = os.path.join(tabllm_dir, f"templates_{dataset_name}.yaml")
    if not os.path.exists(template_path):
        result['valid'] = False
        result['missing_files'].append(template_path)
        result['errors'].append(f"TabLLM template file not found: {template_path}")
    
    # Check notes file
    notes_path = os.path.join(tabllm_dir, "notes", f"notes_{dataset_name}.jsonl")
    if not os.path.exists(notes_path):
        result['warnings'].append(f"TabLLM notes file not found: {notes_path}")
    
    return result


def validate_jolt_metadata(openml_task_id: int, feature_count: Optional[int] = None) -> Dict[str, Any]:
    """
    Validate JOLT metadata for a given OpenML task.
    
    Args:
        openml_task_id: OpenML task ID to validate
        feature_count: Expected number of features (optional)
        
    Returns:
        Dictionary with validation results:
        - 'valid': bool - whether metadata is valid
        - 'missing_files': list - list of missing required files
        - 'errors': list - list of validation errors
        - 'warnings': list - list of warnings
        - 'dataset_name': str - mapped dataset name if found
    """
    project_root = get_project_root()
    jolt_dir = os.path.join(project_root, "examples", "tabular", "llm_baselines", "jolt")
    
    result = {
        'valid': True,
        'missing_files': [],
        'errors': [],
        'warnings': [],
        'dataset_name': None
    }
    
    # Check OpenML task mapping
    mapping_path = os.path.join(jolt_dir, "openml_task_mapping.json")
    if not os.path.exists(mapping_path):
        result['valid'] = False
        result['missing_files'].append(mapping_path)
        result['errors'].append("OpenML task mapping file not found")
        return result
    
    try:
        with open(mapping_path, 'r') as f:
            task_mapping = json.load(f)
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Error loading OpenML task mapping: {e}")
        return result
    
    # Find dataset name for this task ID
    dataset_name = None
    for name, task_id in task_mapping.items():
        if task_id == openml_task_id:
            dataset_name = name
            break
    
    if dataset_name is None:
        result['valid'] = False
        result['errors'].append(f"No JOLT config found for OpenML task ID {openml_task_id}")
        return result
    
    result['dataset_name'] = dataset_name
    
    # Check JOLT config file
    jolt_config_path = os.path.join(jolt_dir, f"jolt_config_{dataset_name}.json")
    if not os.path.exists(jolt_config_path):
        result['valid'] = False
        result['missing_files'].append(jolt_config_path)
        result['errors'].append(f"JOLT config file not found: {jolt_config_path}")
        return result
    
    return result


def validate_metadata_for_models(openml_task_id: int, models: List[str], feature_count: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """
    Validate metadata for multiple models for a given OpenML task.
    
    Args:
        openml_task_id: OpenML task ID to validate
        models: List of model names to validate ('tabllm', 'jolt')
        feature_count: Expected number of features (optional)
        
    Returns:
        Dictionary mapping model names to validation results
    """
    results = {}
    
    for model in models:
        model_lower = model.lower()
        if model_lower == 'tabllm':
            results[model] = validate_tabllm_metadata(openml_task_id, feature_count)
        elif model_lower == 'jolt':
            results[model] = validate_jolt_metadata(openml_task_id, feature_count)
        else:
            results[model] = {
                'valid': True,  # Non-metadata models are always "valid"
                'missing_files': [],
                'errors': [],
                'warnings': [f"Model {model} does not require metadata validation"],
                'dataset_name': None
            }
    
    return results


def generate_metadata_coverage_report(task_ids: List[int] = None, models: List[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive metadata coverage report.
    
    Args:
        task_ids: List of OpenML task IDs to check (if None, uses all from mappings)
        models: List of models to check (default: ['tabllm', 'jolt'])
        
    Returns:
        Dictionary with coverage statistics and detailed results
    """
    if models is None:
        models = ['tabllm', 'jolt']
    
    project_root = get_project_root()
    
    # Get all available task IDs if not provided
    if task_ids is None:
        task_ids = set()
        
        # Get TabLLM task IDs
        tabllm_mapping_path = os.path.join(project_root, "examples", "tabular", "llm_baselines", "tabllm_like", "openml_task_mapping.json")
        if os.path.exists(tabllm_mapping_path):
            try:
                with open(tabllm_mapping_path, 'r') as f:
                    tabllm_mapping = json.load(f)
                task_ids.update(tabllm_mapping.values())
            except Exception as e:
                logger.warning(f"Error reading TabLLM mapping: {e}")
        
        # Get JOLT task IDs
        jolt_mapping_path = os.path.join(project_root, "examples", "tabular", "llm_baselines", "jolt", "openml_task_mapping.json")
        if os.path.exists(jolt_mapping_path):
            try:
                with open(jolt_mapping_path, 'r') as f:
                    jolt_mapping = json.load(f)
                task_ids.update(jolt_mapping.values())
            except Exception as e:
                logger.warning(f"Error reading JOLT mapping: {e}")
        
        task_ids = sorted(list(task_ids))
    
    report = {
        'summary': {
            'total_tasks': len(task_ids),
            'models_checked': models,
            'coverage_by_model': {},
            'tasks_with_full_coverage': 0,
            'tasks_with_partial_coverage': 0,
            'tasks_with_no_coverage': 0
        },
        'detailed_results': {}
    }
    
    # Initialize coverage counters
    for model in models:
        report['summary']['coverage_by_model'][model] = {
            'valid': 0,
            'invalid': 0,
            'coverage_percentage': 0.0
        }
    
    # Check each task
    full_coverage_count = 0
    partial_coverage_count = 0
    no_coverage_count = 0
    
    for task_id in task_ids:
        task_results = validate_metadata_for_models(task_id, models)
        report['detailed_results'][task_id] = task_results
        
        # Count valid models for this task
        valid_models = sum(1 for result in task_results.values() if result['valid'])
        
        if valid_models == len(models):
            full_coverage_count += 1
        elif valid_models > 0:
            partial_coverage_count += 1
        else:
            no_coverage_count += 1
        
        # Update per-model counters
        for model in models:
            if task_results[model]['valid']:
                report['summary']['coverage_by_model'][model]['valid'] += 1
            else:
                report['summary']['coverage_by_model'][model]['invalid'] += 1
    
    # Calculate percentages
    for model in models:
        total = report['summary']['coverage_by_model'][model]['valid'] + report['summary']['coverage_by_model'][model]['invalid']
        if total > 0:
            report['summary']['coverage_by_model'][model]['coverage_percentage'] = (
                report['summary']['coverage_by_model'][model]['valid'] / total * 100
            )
    
    # Update summary counts
    report['summary']['tasks_with_full_coverage'] = full_coverage_count
    report['summary']['tasks_with_partial_coverage'] = partial_coverage_count
    report['summary']['tasks_with_no_coverage'] = no_coverage_count
    
    return report


def print_metadata_coverage_report(report: Dict[str, Any]) -> None:
    """
    Print a formatted metadata coverage report.
    
    Args:
        report: Report dictionary from generate_metadata_coverage_report
    """
    summary = report['summary']
    
    print("="*60)
    print("METADATA COVERAGE REPORT")
    print("="*60)
    print(f"Total tasks checked: {summary['total_tasks']}")
    print(f"Models checked: {', '.join(summary['models_checked'])}")
    print()
    
    print("Coverage by Model:")
    for model, stats in summary['coverage_by_model'].items():
        print(f"  {model.upper()}: {stats['valid']}/{stats['valid'] + stats['invalid']} "
              f"({stats['coverage_percentage']:.1f}%)")
    
    print()
    print("Overall Coverage:")
    print(f"  Tasks with full coverage: {summary['tasks_with_full_coverage']}")
    print(f"  Tasks with partial coverage: {summary['tasks_with_partial_coverage']}")
    print(f"  Tasks with no coverage: {summary['tasks_with_no_coverage']}")
    
    # Show some examples of missing metadata
    print()
    print("Examples of Missing Metadata:")
    count = 0
    for task_id, results in report['detailed_results'].items():
        if count >= 5:  # Limit to 5 examples
            break
        
        for model, result in results.items():
            if not result['valid'] and result['errors']:
                print(f"  Task {task_id} ({model}): {result['errors'][0]}")
                count += 1
                break
    
    print("="*60)