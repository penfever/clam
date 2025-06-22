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
CACHE_DIR = Path.home() / ".clam" / "cache"
MAPPING_CACHE_FILE = CACHE_DIR / "openml_mappings.pkl"
CC18_TASKS_CACHE_FILE = CACHE_DIR / "openml_cc18_tasks.json"

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_openml_cc18_mapping() -> Dict[int, Dict[str, Any]]:
    """
    Get comprehensive mapping of OpenML CC18 tasks with task_id, dataset_id, and dataset_name.
    
    The centralized cache invalidation system automatically handles cache invalidation
    when the data directory changes, so this function focuses on cache loading/saving.
    
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
    logger.info("Creating OpenML mapping from scratch (cache miss or invalidated)")
    mapping = _create_openml_mapping()
    
    # Save to cache
    try:
        with open(MAPPING_CACHE_FILE, 'wb') as f:
            pickle.dump(mapping, f)
        logger.info(f"Saved OpenML mapping to cache: {len(mapping)} tasks")
    except Exception as e:
        logger.warning(f"Failed to save mapping to cache: {e}")
    
    return mapping

def _discover_tasks_from_data_directory() -> Dict[int, Dict[str, Any]]:
    """
    Discover OpenML task IDs from JSON files in the data directory.
    
    Returns:
        Dictionary mapping task_id to basic task information
    """
    import json
    from pathlib import Path
    
    discovered_tasks = {}
    
    # Get the project root and data directory
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent  # Go up to project root
    data_dir = project_root / "data"
    
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return discovered_tasks
    
    # Find all JSON files in data directory and subdirectories
    json_files = list(data_dir.rglob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in data directory")
    
    for json_file in json_files:
        # Try to extract task ID from filename (common pattern is <task_id>.json)
        filename = json_file.stem
        try:
            # Check if filename is a number (task ID)
            if filename.isdigit():
                task_id = int(filename)
                
                # Try to load the JSON file to get metadata
                try:
                    with open(json_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Extract relevant information
                    dataset_name = metadata.get('dataset_name', metadata.get('dataset', f'task_{task_id}'))
                    dataset_id = metadata.get('dataset_id')
                    target_attr = metadata.get('target_variable', {}).get('name') if isinstance(metadata.get('target_variable'), dict) else metadata.get('target_attribute')
                    
                    # Determine task type
                    task_type = metadata.get('task_type', 'classification')
                    if 'regression' in str(metadata).lower() or task_id > 361000:  # Regression tasks typically have higher IDs
                        task_type = 'regression'
                    
                    discovered_tasks[task_id] = {
                        'dataset_id': dataset_id,
                        'dataset_name': dataset_name,
                        'task_type': task_type,
                        'target_attribute': target_attr,
                        'source_file': str(json_file.relative_to(project_root))
                    }
                    
                    logger.debug(f"Discovered task {task_id}: {dataset_name} ({task_type})")
                    
                except Exception as e:
                    logger.warning(f"Could not parse JSON file {json_file}: {e}")
                    # Still add minimal entry
                    discovered_tasks[task_id] = {
                        'dataset_id': None,
                        'dataset_name': f'task_{task_id}',
                        'task_type': 'unknown',
                        'target_attribute': None,
                        'source_file': str(json_file.relative_to(project_root))
                    }
                    
        except ValueError:
            # Filename is not a number, skip
            continue
    
    return discovered_tasks

def _create_openml_mapping() -> Dict[int, Dict[str, Any]]:
    """
    Create OpenML mapping by:
    1. Starting with fallback mapping (includes hardcoded tasks)
    2. Discovering task IDs from data directory JSON files
    3. Optionally querying OpenML API for additional details
    
    Returns:
        Dictionary mapping task_id to task information
    """
    # Start with fallback mapping which includes our manually added tasks
    mapping = _get_fallback_mapping()
    logger.info(f"Starting with {len(mapping)} tasks from fallback mapping")
    
    # Discover tasks from data directory
    discovered_tasks = _discover_tasks_from_data_directory()
    logger.info(f"Discovered {len(discovered_tasks)} unique task IDs from data directory")
    
    # Add discovered tasks to mapping if not already present
    for task_id, metadata in discovered_tasks.items():
        if task_id not in mapping:
            mapping[task_id] = metadata
            logger.debug(f"Added discovered task {task_id}: {metadata.get('dataset_name', 'unknown')}")
        else:
            # Update existing entry with discovered metadata if it has more info
            if metadata.get('dataset_id') and not mapping[task_id].get('dataset_id'):
                mapping[task_id]['dataset_id'] = metadata['dataset_id']
            if metadata.get('source_file'):
                mapping[task_id]['source_file'] = metadata['source_file']
    
    # Optionally try to enrich with OpenML API data
    try:
        import openml
        logger.info("Attempting to enrich mapping with OpenML API data")
        
        # Try to get CC18 study for additional tasks
        try:
            suite = openml.study.get_suite(99)  # 99 is the ID for CC18
            cc18_task_ids = suite.tasks
            logger.info(f"Found {len(cc18_task_ids)} CC18 tasks from OpenML")
            
            # Add CC18 tasks if not already in mapping
            for task_id in cc18_task_ids:
                if task_id not in mapping:
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
                        logger.debug(f"Added CC18 task {task_id}: {dataset.name}")
                    except Exception as e:
                        logger.warning(f"Failed to get details for CC18 task {task_id}: {e}")
                        
        except Exception as e:
            logger.warning(f"Could not fetch CC18 tasks from OpenML: {e}")
            
    except ImportError:
        logger.info("OpenML not available, skipping API enrichment")
    
    logger.info(f"Final mapping contains {len(mapping)} tasks")
    return mapping

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
        
        # Regression tasks from 2025 regression suite
        361085: {'dataset_id': 44145, 'dataset_name': 'sulfur', 'task_type': 'regression', 'target_attribute': 'y1'},
        361086: {'dataset_id': 44146, 'dataset_name': 'medical_charges', 'task_type': 'regression', 'target_attribute': 'AverageTotalPayments'},
        361087: {'dataset_id': 44147, 'dataset_name': 'house_16H', 'task_type': 'regression', 'target_attribute': 'price'},
        361088: {'dataset_id': 44148, 'dataset_name': 'boston', 'task_type': 'regression', 'target_attribute': 'medv'},
        361099: {'dataset_id': 44159, 'dataset_name': 'wine_quality', 'task_type': 'regression', 'target_attribute': 'quality'},
        361103: {'dataset_id': 44068, 'dataset_name': 'pm10', 'task_type': 'regression', 'target_attribute': 'PM.sub.10..sub..particulate.matter..Hourly.measured.'},
        361104: {'dataset_id': 44069, 'dataset_name': 'SGEMM_GPU_kernel_performance', 'task_type': 'regression', 'target_attribute': 'Run1..ms.'},
        363370: {'dataset_id': 44630, 'dataset_name': 'cpu_act', 'task_type': 'regression', 'target_attribute': 'usr'},
        363371: {'dataset_id': 44631, 'dataset_name': 'pol', 'task_type': 'regression', 'target_attribute': 'target'},
        363372: {'dataset_id': 44632, 'dataset_name': 'elevators', 'task_type': 'regression', 'target_attribute': 'goal'},
        363373: {'dataset_id': 44633, 'dataset_name': 'california', 'task_type': 'regression', 'target_attribute': 'median_house_value'},
        363374: {'dataset_id': 44634, 'dataset_name': 'houses', 'task_type': 'regression', 'target_attribute': 'price'},
        363375: {'dataset_id': 44635, 'dataset_name': 'house_8L', 'task_type': 'regression', 'target_attribute': 'price'},
        363376: {'dataset_id': 44636, 'dataset_name': 'diamonds', 'task_type': 'regression', 'target_attribute': 'price'},
        363377: {'dataset_id': 44637, 'dataset_name': 'Brazilian_houses', 'task_type': 'regression', 'target_attribute': 'price'},
        363387: {'dataset_id': 44647, 'dataset_name': 'Bike_Sharing_Demand', 'task_type': 'regression', 'target_attribute': 'count'},
        363388: {'dataset_id': 44648, 'dataset_name': 'nyc-taxi-green-dec-2016', 'task_type': 'regression', 'target_attribute': 'tip_amount'},
        363389: {'dataset_id': 44649, 'dataset_name': 'delays_zurich_transport', 'task_type': 'regression', 'target_attribute': 'arr_delay'},
        363391: {'dataset_id': 44651, 'dataset_name': 'Allstate_Claims_Severity', 'task_type': 'regression', 'target_attribute': 'loss'},
        363394: {'dataset_id': 44654, 'dataset_name': 'diamonds', 'task_type': 'regression', 'target_attribute': 'price'},
        363396: {'dataset_id': 44656, 'dataset_name': 'Mercedes_Benz_Greener_Manufacturing', 'task_type': 'regression', 'target_attribute': 'y'},
        363397: {'dataset_id': 44657, 'dataset_name': 'house_sales', 'task_type': 'regression', 'target_attribute': 'price'},
        363399: {'dataset_id': 44659, 'dataset_name': 'MiamiHousing2016', 'task_type': 'regression', 'target_attribute': 'SALE_PRC'},
        363417: {'dataset_id': 44677, 'dataset_name': 'superconduct', 'task_type': 'regression', 'target_attribute': 'critical_temp'},
        363418: {'dataset_id': 44678, 'dataset_name': 'wine_quality', 'task_type': 'regression', 'target_attribute': 'quality'},
        363426: {'dataset_id': 44686, 'dataset_name': 'black_friday', 'task_type': 'regression', 'target_attribute': 'Purchase'},
        363431: {'dataset_id': 44691, 'dataset_name': 'medical_charges_nominal', 'task_type': 'regression', 'target_attribute': 'AverageTotalPayments'},
        363432: {'dataset_id': 44692, 'dataset_name': 'particulate-matter-ukair-2017', 'task_type': 'regression', 'target_attribute': 'PM.sub.10..sub..particulate.matter..Hourly.measured.'},
        363433: {'dataset_id': 44693, 'dataset_name': 'diamonds', 'task_type': 'regression', 'target_attribute': 'price'},
        363434: {'dataset_id': 44694, 'dataset_name': 'OnlineNewsPopularity', 'task_type': 'regression', 'target_attribute': 'shares'},
        363435: {'dataset_id': 44695, 'dataset_name': 'Buzzinsocialmedia_Twitter', 'task_type': 'regression', 'target_attribute': 'Mean..'},
        363436: {'dataset_id': 44696, 'dataset_name': 'SeoulBikeData', 'task_type': 'regression', 'target_attribute': 'Rented.Bike.Count'},
        363437: {'dataset_id': 44697, 'dataset_name': 'Moneyball', 'task_type': 'regression', 'target_attribute': 'TARGET_WINS'},
        363438: {'dataset_id': 44698, 'dataset_name': 'topo_2_1', 'task_type': 'regression', 'target_attribute': 'y'},
        363439: {'dataset_id': 44699, 'dataset_name': 'visualizing_soil', 'task_type': 'regression', 'target_attribute': 'Depth'},
        363440: {'dataset_id': 44700, 'dataset_name': 'QSAR-TID-10980', 'task_type': 'regression', 'target_attribute': 'P1'},
        363442: {'dataset_id': 44702, 'dataset_name': 'QSAR-TID-11', 'task_type': 'regression', 'target_attribute': 'P1'},
        363443: {'dataset_id': 44703, 'dataset_name': 'physiochemical_protein', 'task_type': 'regression', 'target_attribute': 'RMSD'},
        363444: {'dataset_id': 44704, 'dataset_name': 'tecator', 'task_type': 'regression', 'target_attribute': 'Fat'},
        363447: {'dataset_id': 44707, 'dataset_name': 'space_ga', 'task_type': 'regression', 'target_attribute': 'price'},
        363448: {'dataset_id': 44708, 'dataset_name': 'Yolanda', 'task_type': 'regression', 'target_attribute': 'class'},
        363452: {'dataset_id': 44712, 'dataset_name': 'college_math', 'task_type': 'regression', 'target_attribute': 'G3'},
        363453: {'dataset_id': 44713, 'dataset_name': 'fertility_diagnosis', 'task_type': 'regression', 'target_attribute': 'output'},
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