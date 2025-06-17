#!/usr/bin/env python
"""
OpenML mapping utilities for task_id, dataset_id, and dataset_name relationships.

This module provides functions to retrieve and cache the mappings between OpenML task IDs,
dataset IDs, and dataset names. Used primarily for imputing missing task_id values in
wandb analysis when only dataset names are available.
"""

import os
import json
import logging
from typing import Dict, Optional, Tuple, Any
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache file for OpenML mappings
CACHE_DIR = Path.home() / ".llata" / "cache"
MAPPING_CACHE_FILE = CACHE_DIR / "openml_mappings.pkl"
CC18_TASKS_CACHE_FILE = CACHE_DIR / "openml_cc18_tasks.json"

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_openml_cc18_mapping() -> Dict[int, Dict[str, Any]]:
    """
    Get comprehensive mapping of OpenML CC18 tasks with task_id, dataset_id, and dataset_name.
    
    Returns:
        Dictionary mapping task_id to {'dataset_id': int, 'dataset_name': str, 'num_classes': int, etc.}
    """
    # Try to load from cache first
    if MAPPING_CACHE_FILE.exists():
        try:
            with open(MAPPING_CACHE_FILE, 'rb') as f:
                cached_mapping = pickle.load(f)
                logger.debug(f"Loaded OpenML mapping from cache: {len(cached_mapping)} tasks")
                return cached_mapping
        except Exception as e:
            logger.warning(f"Failed to load cached mapping: {e}")
    
    # If cache doesn't exist or failed to load, create new mapping
    logger.info("Creating OpenML CC18 mapping from scratch")
    mapping = _create_openml_mapping()
    
    # Save to cache
    try:
        with open(MAPPING_CACHE_FILE, 'wb') as f:
            pickle.dump(mapping, f)
        logger.info(f"Saved OpenML mapping to cache: {len(mapping)} tasks")
    except Exception as e:
        logger.warning(f"Failed to save mapping to cache: {e}")
    
    return mapping

def _create_openml_mapping() -> Dict[int, Dict[str, Any]]:
    """
    Create OpenML mapping by querying the API or using fallback data.
    
    Returns:
        Dictionary mapping task_id to task information
    """
    try:
        import openml
        logger.info("Attempting to fetch CC18 tasks from OpenML API")
        
        # Try to get CC18 study
        try:
            suite = openml.study.get_suite(99)  # 99 is the ID for CC18
            task_ids = suite.tasks
        except Exception as e1:
            logger.warning(f"Error using get_suite: {e1}")
            try:
                study = openml.study.functions._get_study(99, entity_type='task')
                task_ids = study.tasks
            except Exception as e2:
                logger.warning(f"Error using get_study: {e2}")
                # Use hardcoded list as fallback
                task_ids = _get_hardcoded_cc18_tasks()
        
        logger.info(f"Retrieved {len(task_ids)} task IDs from CC18")
        
        # Get detailed information for each task
        mapping = {}
        for task_id in task_ids:
            try:
                task = openml.tasks.get_task(task_id)
                dataset = task.get_dataset()
                
                mapping[task_id] = {
                    'dataset_id': task.dataset_id,
                    'dataset_name': dataset.name,
                    'num_classes': len(task.class_labels) if hasattr(task, 'class_labels') else None,
                    'num_features': len(dataset.features) if hasattr(dataset, 'features') and isinstance(dataset.features, dict) else None,
                    'target_attribute': task.target_name if hasattr(task, 'target_name') else None
                }
                
                logger.debug(f"Task {task_id}: {dataset.name} (dataset_id: {task.dataset_id})")
                
            except Exception as e:
                logger.warning(f"Failed to get details for task {task_id}: {e}")
                # Add minimal entry
                mapping[task_id] = {
                    'dataset_id': None,
                    'dataset_name': f"task_{task_id}",
                    'num_classes': None,
                    'num_features': None,
                    'target_attribute': None
                }
        
        logger.info(f"Successfully created mapping for {len(mapping)} tasks")
        return mapping
        
    except ImportError:
        logger.warning("OpenML not available, using fallback mapping")
        return _get_fallback_mapping()
    except Exception as e:
        logger.error(f"Error creating OpenML mapping: {e}")
        return _get_fallback_mapping()

def _get_hardcoded_cc18_tasks() -> list:
    """Get hardcoded list of CC18 task IDs as fallback."""
    return [
        3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54, 151, 182, 188, 38, 307, 300, 458, 
        469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1590, 4134, 1489, 1494, 1497, 1501, 1485, 1486, 1487, 1468, 
        1475, 1462, 4534, 1461, 4538, 1478, 40668, 40966, 40982, 40983, 40975, 40984, 40996, 41027, 23517, 40978, 
        40670, 40701
    ]

def _get_fallback_mapping() -> Dict[int, Dict[str, Any]]:
    """
    Get fallback mapping with known OpenML CC18 relationships.
    This is a curated list based on common CC18 datasets.
    """
    return {
        # Major CC18 datasets with known mappings
        3: {'dataset_id': 3, 'dataset_name': 'kr-vs-kp', 'num_classes': 2, 'num_features': 36, 'target_attribute': 'class'},
        6: {'dataset_id': 6, 'dataset_name': 'letter', 'num_classes': 26, 'num_features': 16, 'target_attribute': 'class'},
        11: {'dataset_id': 11, 'dataset_name': 'balance-scale', 'num_classes': 3, 'num_features': 4, 'target_attribute': 'class'},
        12: {'dataset_id': 12, 'dataset_name': 'mfeat-morphological', 'num_classes': 10, 'num_features': 6, 'target_attribute': 'class'},
        14: {'dataset_id': 14, 'dataset_name': 'mfeat-karhunen', 'num_classes': 10, 'num_features': 64, 'target_attribute': 'class'},
        15: {'dataset_id': 15, 'dataset_name': 'mfeat-zernike', 'num_classes': 10, 'num_features': 47, 'target_attribute': 'class'},
        16: {'dataset_id': 16, 'dataset_name': 'mfeat-pixel', 'num_classes': 10, 'num_features': 240, 'target_attribute': 'class'},
        18: {'dataset_id': 18, 'dataset_name': 'mfeat-factors', 'num_classes': 10, 'num_features': 216, 'target_attribute': 'class'},
        22: {'dataset_id': 22, 'dataset_name': 'mfeat-fourier', 'num_classes': 10, 'num_features': 76, 'target_attribute': 'class'},
        23: {'dataset_id': 23, 'dataset_name': 'cmc', 'num_classes': 3, 'num_features': 9, 'target_attribute': 'Contraceptive_method_used'},
        28: {'dataset_id': 28, 'dataset_name': 'optdigits', 'num_classes': 10, 'num_features': 64, 'target_attribute': 'class'},
        29: {'dataset_id': 29, 'dataset_name': 'credit-approval', 'num_classes': 2, 'num_features': 15, 'target_attribute': 'class'},
        31: {'dataset_id': 31, 'dataset_name': 'credit-g', 'num_classes': 2, 'num_features': 20, 'target_attribute': 'class'},
        32: {'dataset_id': 32, 'dataset_name': 'pendigits', 'num_classes': 10, 'num_features': 16, 'target_attribute': 'class'},
        37: {'dataset_id': 37, 'dataset_name': 'diabetes', 'num_classes': 2, 'num_features': 8, 'target_attribute': 'class'},
        44: {'dataset_id': 44, 'dataset_name': 'spambase', 'num_classes': 2, 'num_features': 57, 'target_attribute': 'class'},
        46: {'dataset_id': 46, 'dataset_name': 'splice', 'num_classes': 3, 'num_features': 60, 'target_attribute': 'class'},
        50: {'dataset_id': 50, 'dataset_name': 'tic-tac-toe', 'num_classes': 2, 'num_features': 9, 'target_attribute': 'class'},
        54: {'dataset_id': 54, 'dataset_name': 'vehicle', 'num_classes': 4, 'num_features': 18, 'target_attribute': 'class'},
        151: {'dataset_id': 151, 'dataset_name': 'electricity', 'num_classes': 2, 'num_features': 8, 'target_attribute': 'class'},
        182: {'dataset_id': 182, 'dataset_name': 'satimage', 'num_classes': 6, 'num_features': 36, 'target_attribute': 'class'},
        188: {'dataset_id': 188, 'dataset_name': 'eucalyptus', 'num_classes': 5, 'num_features': 19, 'target_attribute': 'Utility'},
        38: {'dataset_id': 38, 'dataset_name': 'sick', 'num_classes': 2, 'num_features': 29, 'target_attribute': 'Class'},
        307: {'dataset_id': 307, 'dataset_name': 'vowel', 'num_classes': 11, 'num_features': 13, 'target_attribute': 'Class'},
        300: {'dataset_id': 300, 'dataset_name': 'isolet', 'num_classes': 26, 'num_features': 617, 'target_attribute': 'class'},
        458: {'dataset_id': 458, 'dataset_name': 'analcatdata_authorship', 'num_classes': 4, 'num_features': 70, 'target_attribute': 'class'},
        469: {'dataset_id': 469, 'dataset_name': 'analcatdata_dmft', 'num_classes': 6, 'num_features': 4, 'target_attribute': 'class'},
        554: {'dataset_id': 554, 'dataset_name': 'mnist_784', 'num_classes': 10, 'num_features': 784, 'target_attribute': 'class'},
        1049: {'dataset_id': 1049, 'dataset_name': 'pc4', 'num_classes': 2, 'num_features': 37, 'target_attribute': 'c'},
        1050: {'dataset_id': 1050, 'dataset_name': 'pc3', 'num_classes': 2, 'num_features': 37, 'target_attribute': 'c'},
        1053: {'dataset_id': 1053, 'dataset_name': 'jm1', 'num_classes': 2, 'num_features': 21, 'target_attribute': 'defects'},
        1063: {'dataset_id': 1063, 'dataset_name': 'kc2', 'num_classes': 2, 'num_features': 21, 'target_attribute': 'problems'},
        1067: {'dataset_id': 1067, 'dataset_name': 'kc1', 'num_classes': 2, 'num_features': 21, 'target_attribute': 'defects'},
        1068: {'dataset_id': 1068, 'dataset_name': 'pc1', 'num_classes': 2, 'num_features': 21, 'target_attribute': 'c'},
        1590: {'dataset_id': 1590, 'dataset_name': 'adult', 'num_classes': 2, 'num_features': 14, 'target_attribute': 'class'},
        4134: {'dataset_id': 4134, 'dataset_name': 'bioresponse', 'num_classes': 2, 'num_features': 1776, 'target_attribute': 'target'},
        1489: {'dataset_id': 1489, 'dataset_name': 'phoneme', 'num_classes': 2, 'num_features': 5, 'target_attribute': 'class'},
        1494: {'dataset_id': 1494, 'dataset_name': 'a9a', 'num_classes': 2, 'num_features': 123, 'target_attribute': 'class'},
        1497: {'dataset_id': 1497, 'dataset_name': 'w8a', 'num_classes': 2, 'num_features': 300, 'target_attribute': 'class'},
        1501: {'dataset_id': 1501, 'dataset_name': 'phishing', 'num_classes': 2, 'num_features': 30, 'target_attribute': 'Result'},
        1485: {'dataset_id': 1485, 'dataset_name': 'madelon', 'num_classes': 2, 'num_features': 500, 'target_attribute': 'class'},
        1486: {'dataset_id': 1486, 'dataset_name': 'nomao', 'num_classes': 2, 'num_features': 118, 'target_attribute': 'Class'},
        1487: {'dataset_id': 1487, 'dataset_name': 'connect-4', 'num_classes': 3, 'num_features': 42, 'target_attribute': 'class'},
        1468: {'dataset_id': 1468, 'dataset_name': 'cnae-9', 'num_classes': 9, 'num_features': 856, 'target_attribute': 'class'},
        1475: {'dataset_id': 1475, 'dataset_name': 'first-order-theorem', 'num_classes': 6, 'num_features': 51, 'target_attribute': 'class'},
        1462: {'dataset_id': 1462, 'dataset_name': 'banknote', 'num_classes': 2, 'num_features': 4, 'target_attribute': 'class'},
        4534: {'dataset_id': 4534, 'dataset_name': 'segment', 'num_classes': 7, 'num_features': 19, 'target_attribute': 'class'},
        1461: {'dataset_id': 1461, 'dataset_name': 'bank-marketing', 'num_classes': 2, 'num_features': 16, 'target_attribute': 'y'},
        4538: {'dataset_id': 4538, 'dataset_name': 'vehicle', 'num_classes': 4, 'num_features': 18, 'target_attribute': 'class'},
        1478: {'dataset_id': 1478, 'dataset_name': 'har', 'num_classes': 6, 'num_features': 561, 'target_attribute': 'class'},
        40668: {'dataset_id': 40668, 'dataset_name': 'blood-transfusion', 'num_classes': 2, 'num_features': 4, 'target_attribute': 'class'},
        40966: {'dataset_id': 40966, 'dataset_name': 'MagicTelescope', 'num_classes': 2, 'num_features': 10, 'target_attribute': 'class'},
        40982: {'dataset_id': 40982, 'dataset_name': 'steel-plates', 'num_classes': 7, 'num_features': 33, 'target_attribute': 'class'},
        40983: {'dataset_id': 40983, 'dataset_name': 'wilt', 'num_classes': 2, 'num_features': 5, 'target_attribute': 'class'},
        40975: {'dataset_id': 40975, 'dataset_name': 'car', 'num_classes': 4, 'num_features': 6, 'target_attribute': 'class'},
        40984: {'dataset_id': 40984, 'dataset_name': 'segment', 'num_classes': 7, 'num_features': 19, 'target_attribute': 'class'},
        40996: {'dataset_id': 40996, 'dataset_name': 'Fashion-MNIST', 'num_classes': 10, 'num_features': 784, 'target_attribute': 'class'},
        41027: {'dataset_id': 41027, 'dataset_name': 'jungle_chess', 'num_classes': 3, 'num_features': 6, 'target_attribute': 'class'},
        23517: {'dataset_id': 23517, 'dataset_name': 'higgs', 'num_classes': 2, 'num_features': 28, 'target_attribute': 'class'},
        40978: {'dataset_id': 40978, 'dataset_name': 'mfeat-zernike', 'num_classes': 10, 'num_features': 47, 'target_attribute': 'class'},
        40670: {'dataset_id': 40670, 'dataset_name': 'dna', 'num_classes': 3, 'num_features': 180, 'target_attribute': 'class'},
        40701: {'dataset_id': 40701, 'dataset_name': 'churn', 'num_classes': 2, 'num_features': 20, 'target_attribute': 'class'},
    }

def impute_task_id_from_dataset_name(dataset_name: str) -> Optional[int]:
    """
    Impute task_id from dataset_name using OpenML mapping.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        task_id if found, None otherwise
    """
    if not dataset_name:
        return None
    
    mapping = get_openml_cc18_mapping()
    
    # Direct name match
    for task_id, info in mapping.items():
        if info['dataset_name'] == dataset_name:
            logger.debug(f"Found exact match: {dataset_name} -> task_id {task_id}")
            return task_id
    
    # Fuzzy matching for common variations
    dataset_name_clean = dataset_name.lower().replace('-', '_').replace(' ', '_')
    for task_id, info in mapping.items():
        info_name_clean = info['dataset_name'].lower().replace('-', '_').replace(' ', '_')
        if info_name_clean == dataset_name_clean:
            logger.debug(f"Found fuzzy match: {dataset_name} -> {info['dataset_name']} -> task_id {task_id}")
            return task_id
    
    logger.debug(f"No task_id found for dataset_name: {dataset_name}")
    return None

def impute_task_id_from_dataset_id(dataset_id: int) -> Optional[int]:
    """
    Impute task_id from dataset_id using OpenML mapping.
    
    Args:
        dataset_id: OpenML dataset ID
        
    Returns:
        task_id if found, None otherwise
    """
    if not dataset_id:
        return None
    
    mapping = get_openml_cc18_mapping()
    
    for task_id, info in mapping.items():
        if info['dataset_id'] == dataset_id:
            logger.debug(f"Found match: dataset_id {dataset_id} -> task_id {task_id}")
            return task_id
    
    logger.debug(f"No task_id found for dataset_id: {dataset_id}")
    return None

def get_task_info(task_id: int) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive information about a task.
    
    Args:
        task_id: OpenML task ID
        
    Returns:
        Dictionary with task information or None if not found
    """
    mapping = get_openml_cc18_mapping()
    return mapping.get(task_id)

def clear_cache():
    """Clear all cached OpenML mapping data."""
    try:
        if MAPPING_CACHE_FILE.exists():
            MAPPING_CACHE_FILE.unlink()
        if CC18_TASKS_CACHE_FILE.exists():
            CC18_TASKS_CACHE_FILE.unlink()
        logger.info("Cleared OpenML mapping cache")
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")

if __name__ == "__main__":
    # Test the mapping functionality
    logging.basicConfig(level=logging.INFO)
    
    mapping = get_openml_cc18_mapping()
    print(f"Loaded mapping for {len(mapping)} tasks")
    
    # Test imputation
    test_cases = [
        ("kr-vs-kp", 3),
        ("adult", 1590),
        ("electricity", 151),
        ("nonexistent", None)
    ]
    
    for dataset_name, expected_task_id in test_cases:
        result = impute_task_id_from_dataset_name(dataset_name)
        status = "✅" if result == expected_task_id else "❌"
        print(f"{status} {dataset_name} -> {result} (expected {expected_task_id})")