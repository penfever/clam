#!/usr/bin/env python
"""
Task type detection utilities for determining whether a dataset represents
a classification or regression task.

This module provides functions to automatically detect task types from:
1. OpenML task metadata
2. Dataset characteristics (target variable analysis)
3. Manual specifications
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, Optional, Dict, Any, Tuple
from sklearn.utils.multiclass import type_of_target
import warnings

logger = logging.getLogger(__name__)

def detect_task_type_from_openml_task(task) -> Optional[str]:
    """
    Detect task type from OpenML task object.
    
    Args:
        task: OpenML task object
        
    Returns:
        'classification' or 'regression' if detected, None if uncertain
    """
    try:
        # Try to get task type directly from OpenML task metadata
        if hasattr(task, 'task_type') and task.task_type:
            task_type_str = str(task.task_type).lower()
            if 'classif' in task_type_str:
                logger.debug(f"Detected classification from OpenML task.task_type: {task.task_type}")
                return 'classification'
            elif 'regress' in task_type_str:
                logger.debug(f"Detected regression from OpenML task.task_type: {task.task_type}")
                return 'regression'
        
        # Try to get task type from task type ID
        if hasattr(task, 'task_type_id'):
            # OpenML task type IDs: 1=classification, 2=regression, etc.
            if task.task_type_id == 1:
                logger.debug(f"Detected classification from OpenML task_type_id: {task.task_type_id}")
                return 'classification'
            elif task.task_type_id == 2:
                logger.debug(f"Detected regression from OpenML task_type_id: {task.task_type_id}")
                return 'regression'
        
        # Check if the task has class labels (classification indicator)
        if hasattr(task, 'class_labels') and task.class_labels:
            logger.debug(f"Detected classification from OpenML class_labels: {len(task.class_labels)} classes")
            return 'classification'
        
        logger.debug("Could not determine task type from OpenML task metadata")
        return None
        
    except Exception as e:
        logger.warning(f"Error detecting task type from OpenML task: {e}")
        return None

def detect_task_type_from_target(y: Union[np.ndarray, pd.Series, list], 
                                target_name: Optional[str] = None,
                                classification_threshold: int = 50) -> str:
    """
    Detect task type by analyzing the target variable.
    
    Args:
        y: Target variable values
        target_name: Name of target variable (for logging)
        classification_threshold: Maximum unique values for classification
        
    Returns:
        'classification' or 'regression'
    """
    try:
        # Convert to numpy array for analysis
        if hasattr(y, 'values'):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Remove any NaN values for analysis
        y_clean = y_array[~pd.isna(y_array)]
        
        if len(y_clean) == 0:
            logger.warning("Target variable contains only NaN values, defaulting to classification")
            return 'classification'
        
        # Use sklearn's type_of_target for initial detection
        try:
            target_type = type_of_target(y_clean)
            logger.debug(f"sklearn type_of_target result: {target_type}")
            
            if target_type in ['binary', 'multiclass', 'multilabel-indicator']:
                logger.info(f"Detected classification task (sklearn type: {target_type})")
                return 'classification'
            elif target_type == 'continuous':
                logger.info(f"Detected regression task (sklearn type: {target_type})")
                return 'regression'
        except Exception as e:
            logger.debug(f"sklearn type_of_target failed: {e}, falling back to manual analysis")
        
        # Manual analysis as fallback
        unique_values = np.unique(y_clean)
        num_unique = len(unique_values)
        
        # Check data types
        is_numeric = np.issubdtype(y_clean.dtype, np.number)
        is_integer_like = False
        
        if is_numeric:
            # Check if all values are integer-like (even if stored as float)
            is_integer_like = np.allclose(y_clean, np.round(y_clean), rtol=1e-10, atol=1e-10)
        
        # Apply heuristics
        if not is_numeric:
            # Non-numeric targets are always classification
            logger.info(f"Detected classification: non-numeric target with {num_unique} unique values")
            return 'classification'
        
        if num_unique <= 2:
            # Binary classification
            logger.info(f"Detected classification: binary target with values {unique_values}")
            return 'classification'
        
        if is_integer_like and num_unique <= classification_threshold:
            # Small number of integer values suggests classification
            logger.info(f"Detected classification: {num_unique} unique integer-like values (≤ {classification_threshold})")
            return 'classification'
        
        if not is_integer_like or num_unique > classification_threshold:
            # Many unique values or non-integer values suggest regression
            logger.info(f"Detected regression: {num_unique} unique values, integer-like: {is_integer_like}")
            return 'regression'
        
        # Default fallback
        logger.warning(f"Uncertain task type for target with {num_unique} unique values, defaulting to classification")
        return 'classification'
        
    except Exception as e:
        logger.error(f"Error analyzing target variable: {e}")
        return 'classification'  # Safe default

def detect_task_type_from_dataset_info(dataset_info: Dict[str, Any]) -> Optional[str]:
    """
    Detect task type from dataset information dictionary.
    
    Args:
        dataset_info: Dictionary containing dataset metadata
        
    Returns:
        'classification' or 'regression' if detected, None if uncertain
    """
    try:
        # Check if task type is explicitly specified
        if 'task_type' in dataset_info:
            task_type = str(dataset_info['task_type']).lower()
            if 'classif' in task_type:
                return 'classification'
            elif 'regress' in task_type:
                return 'regression'
        
        # Check for number of classes indicator
        if 'num_classes' in dataset_info and dataset_info['num_classes']:
            num_classes = dataset_info['num_classes']
            if isinstance(num_classes, (int, float)) and num_classes >= 2:
                logger.debug(f"Detected classification from num_classes: {num_classes}")
                return 'classification'
        
        # Check target attribute name for hints
        if 'target_attribute' in dataset_info and dataset_info['target_attribute']:
            target_name = str(dataset_info['target_attribute']).lower()
            
            # Common classification indicators
            classification_hints = ['class', 'label', 'category', 'type', 'outcome']
            if any(hint in target_name for hint in classification_hints):
                logger.debug(f"Detected classification from target name hints: {target_name}")
                return 'classification'
            
            # Common regression indicators  
            regression_hints = ['price', 'value', 'amount', 'score', 'rating', 'age', 'time', 'count']
            if any(hint in target_name for hint in regression_hints):
                logger.debug(f"Detected regression from target name hints: {target_name}")
                return 'regression'
        
        return None
        
    except Exception as e:
        logger.warning(f"Error detecting task type from dataset info: {e}")
        return None

def detect_task_type(dataset: Optional[Dict[str, Any]] = None,
                    y: Optional[Union[np.ndarray, pd.Series, list]] = None,
                    task: Optional[Any] = None,
                    dataset_info: Optional[Dict[str, Any]] = None,
                    manual_override: Optional[str] = None,
                    classification_threshold: int = 50) -> Tuple[str, str]:
    """
    Comprehensive task type detection using multiple methods.
    
    Args:
        dataset: Dataset dictionary (may contain 'y' key)
        y: Target variable values
        task: OpenML task object
        dataset_info: Dataset metadata dictionary
        manual_override: Manual specification ('classification' or 'regression')
        classification_threshold: Maximum unique values for classification
        
    Returns:
        Tuple of (task_type, detection_method)
        task_type: 'classification' or 'regression'
        detection_method: Description of how task type was determined
    """
    
    # 1. Manual override takes precedence
    if manual_override:
        manual_override = manual_override.lower()
        if manual_override in ['classification', 'regression']:
            logger.info(f"Using manual task type override: {manual_override}")
            return manual_override, "manual_override"
        else:
            logger.warning(f"Invalid manual override '{manual_override}', ignoring")
    
    # 2. Try OpenML task metadata
    if task is not None:
        detected_type = detect_task_type_from_openml_task(task)
        if detected_type:
            logger.info(f"Detected task type from OpenML task: {detected_type}")
            return detected_type, "openml_task"
    
    # 3. Try dataset info metadata
    if dataset_info is not None:
        detected_type = detect_task_type_from_dataset_info(dataset_info)
        if detected_type:
            logger.info(f"Detected task type from dataset info: {detected_type}")
            return detected_type, "dataset_info"
    
    # 4. Extract target variable from dataset if not provided
    if y is None and dataset is not None:
        if 'y' in dataset:
            y = dataset['y']
        elif 'target' in dataset:
            y = dataset['target']
    
    # 5. Analyze target variable
    if y is not None:
        target_name = None
        if dataset and 'target_name' in dataset:
            target_name = dataset['target_name']
        elif dataset_info and 'target_attribute' in dataset_info:
            target_name = dataset_info['target_attribute']
        
        detected_type = detect_task_type_from_target(
            y, target_name, classification_threshold
        )
        logger.info(f"Detected task type from target analysis: {detected_type}")
        return detected_type, "target_analysis"
    
    # 6. Default fallback
    logger.warning("Could not determine task type, defaulting to classification")
    return 'classification', "default_fallback"

def is_regression_task(dataset: Optional[Dict[str, Any]] = None,
                      y: Optional[Union[np.ndarray, pd.Series, list]] = None,
                      task: Optional[Any] = None,
                      dataset_info: Optional[Dict[str, Any]] = None,
                      manual_override: Optional[str] = None) -> bool:
    """
    Determine if the task is a regression task.
    
    Args:
        dataset: Dataset dictionary
        y: Target variable values
        task: OpenML task object
        dataset_info: Dataset metadata dictionary
        manual_override: Manual specification
        
    Returns:
        True if regression task, False if classification task
    """
    task_type, _ = detect_task_type(dataset, y, task, dataset_info, manual_override)
    return task_type == 'regression'

def is_classification_task(dataset: Optional[Dict[str, Any]] = None,
                          y: Optional[Union[np.ndarray, pd.Series, list]] = None,
                          task: Optional[Any] = None,
                          dataset_info: Optional[Dict[str, Any]] = None,
                          manual_override: Optional[str] = None) -> bool:
    """
    Determine if the task is a classification task.
    
    Args:
        dataset: Dataset dictionary
        y: Target variable values
        task: OpenML task object
        dataset_info: Dataset metadata dictionary
        manual_override: Manual specification
        
    Returns:
        True if classification task, False if regression task
    """
    task_type, _ = detect_task_type(dataset, y, task, dataset_info, manual_override)
    return task_type == 'classification'

def get_target_statistics(y: Union[np.ndarray, pd.Series, list]) -> Dict[str, Any]:
    """
    Get statistics about the target variable for context in prompts.
    
    Args:
        y: Target variable values
        
    Returns:
        Dictionary with target statistics
    """
    try:
        # Convert to numpy array
        if hasattr(y, 'values'):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Remove NaN values
        y_clean = y_array[~pd.isna(y_array)]
        
        if len(y_clean) == 0:
            return {'error': 'All values are NaN'}
        
        stats = {
            'count': len(y_clean),
            'unique_count': len(np.unique(y_clean)),
            'dtype': str(y_clean.dtype)
        }
        
        # Add numeric statistics if applicable
        if np.issubdtype(y_clean.dtype, np.number):
            stats.update({
                'min': float(np.min(y_clean)),
                'max': float(np.max(y_clean)),
                'mean': float(np.mean(y_clean)),
                'std': float(np.std(y_clean)),
                'median': float(np.median(y_clean))
            })
            
            # Add percentiles for regression context
            percentiles = np.percentile(y_clean, [25, 75])
            stats.update({
                'q25': float(percentiles[0]),
                'q75': float(percentiles[1])
            })
        else:
            # For non-numeric targets, provide value counts
            unique_vals, counts = np.unique(y_clean, return_counts=True)
            stats['value_counts'] = dict(zip(unique_vals.tolist(), counts.tolist()))
        
        return stats
        
    except Exception as e:
        logger.warning(f"Error computing target statistics: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # Test the task detection functionality
    logging.basicConfig(level=logging.INFO)
    
    # Test cases
    test_cases = [
        # Classification examples
        ([0, 1, 1, 0, 1, 0], "binary classification"),
        ([1, 2, 3, 1, 2, 3, 1, 2], "multiclass classification"),
        (['cat', 'dog', 'cat', 'bird', 'dog'], "string classification"),
        
        # Regression examples
        ([1.5, 2.7, 3.1, 4.6, 5.2], "continuous regression"),
        (np.random.randn(100) * 10 + 50, "normal distribution regression"),
        (list(range(100)), "integer sequence regression"),
    ]
    
    print("Testing task type detection:")
    for y_test, description in test_cases:
        task_type, method = detect_task_type(y=y_test)
        stats = get_target_statistics(y_test)
        print(f"  {description}: {task_type} (method: {method})")
        print(f"    Stats: {stats.get('unique_count', 'N/A')} unique values, "
              f"range: {stats.get('min', 'N/A')}-{stats.get('max', 'N/A')}")