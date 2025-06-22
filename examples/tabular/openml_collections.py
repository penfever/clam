#!/usr/bin/env python
"""
Shared task collection fetchers for OpenML evaluation scripts.

This module provides reusable functionality for fetching OpenML collections
like CC18, regression 2025, etc.
"""

import logging
import openml
from typing import List, Any, Optional

logger = logging.getLogger(__name__)


def get_openml_cc18_tasks() -> List[Any]:
    """
    Get the list of tasks in the OpenML CC18 collection (study_id=99).
    
    Returns:
        List of OpenML task objects
    """
    logger.info("Fetching OpenML CC18 collection (study_id=99)")
    
    try:
        # Try the newer API method first
        logger.info("Attempting to fetch CC18 using newer API method (get_suite)")
        suite = openml.study.get_suite(99)  # 99 is the ID for CC18
        task_ids = suite.tasks
    except Exception as e1:
        logger.warning(f"Error using get_suite: {e1}")
        try:
            # Try fallback method
            logger.info("Attempting fallback method with get_study and entity_type='task'")
            study = openml.study.functions._get_study(99, entity_type='task')
            task_ids = study.tasks
        except Exception as e2:
            logger.warning(f"Error using get_study fallback: {e2}")
            # Hardcoded list of CC18 tasks as a last resort
            logger.info("Using hardcoded list of CC18 tasks")
            task_ids = [
                3573, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3917, 3918,
                3950, 3954, 7592, 7593, 9914, 9946, 9957, 9960, 9961, 9962, 9964, 9965, 9966, 9967, 9968,
                9970, 9971, 9976, 9977, 9978, 9979, 9980, 9981, 9983, 9985, 9986, 9987, 10060, 10061,
                10064, 10065, 10066, 10067, 10068, 10069, 10070, 10071, 10072, 10073, 10074, 10075, 10076,
                10077, 10078, 10079, 10080, 10081, 10082, 10083, 10084, 10085, 10086, 10087, 10088, 10089,
                10090, 10092, 10093, 10096, 10097, 10098, 10099, 10100, 10101, 14954, 14965, 14969, 14970,
                125920, 125921, 125922, 125923, 125928, 125929, 125920, 125921, 125922, 125923, 125928,
                125929, 125930, 125931, 125932, 125933, 125934, 14954, 14965, 14969, 14970, 34536, 34537,
                34539, 146574
            ]
            # Remove duplicates
            task_ids = list(set(task_ids))

    logger.info(f"Retrieved {len(task_ids)} tasks from CC18 collection")
    
    tasks = []
    for task_id in task_ids:
        try:
            task = openml.tasks.get_task(task_id)
            tasks.append(task)
            logger.info(f"Retrieved task {task_id}: {task.get_dataset().name}")
        except Exception as e:
            logger.error(f"Error retrieving task {task_id}: {e}")
    
    logger.info(f"Successfully retrieved {len(tasks)} CC18 tasks")
    return tasks


def get_openml_regression_2025_tasks() -> List[Any]:
    """
    Get the list of tasks in the New OpenML Suite 2025 regression collection (study_id=455).
    
    Returns:
        List of OpenML task objects
    """
    logger.info("Fetching New OpenML Suite 2025 regression collection (study_id=455)")
    
    try:
        # Try the newer API method first
        logger.info("Attempting to fetch regression suite using newer API method (get_suite)")
        suite = openml.study.get_suite(455)  # 455 is the ID for New_OpenML_Suite_2025_regression
        task_ids = suite.tasks
    except Exception as e1:
        logger.warning(f"Error using get_suite: {e1}")
        try:
            # Try fallback method
            logger.info("Attempting fallback method with get_study and entity_type='task'")
            study = openml.study.functions._get_study(455, entity_type='task')
            task_ids = study.tasks
        except Exception as e2:
            logger.warning(f"Error using get_study fallback: {e2}")
            # If both methods fail, we'll need to fetch individual tasks or use a hardcoded list
            logger.error("Could not fetch regression suite. Please check study_id=455 exists and contains regression tasks.")
            return []

    logger.info(f"Retrieved {len(task_ids)} task IDs from regression collection")
    
    tasks = []
    for task_id in task_ids:
        try:
            task = openml.tasks.get_task(task_id)
            # Verify this is a regression task
            if task.task_type.lower() != 'supervised regression':
                logger.warning(f"Task {task_id} is not a regression task (type: {task.task_type}), skipping")
                continue
            tasks.append(task)
            logger.info(f"Retrieved regression task {task_id}: {task.get_dataset().name}")
        except Exception as e:
            logger.error(f"Error retrieving task {task_id}: {e}")
    
    logger.info(f"Successfully retrieved {len(tasks)} regression tasks")
    return tasks


def get_openml_collection_tasks(collection_name: str, study_id: Optional[int] = None) -> List[Any]:
    """
    Get tasks from an OpenML collection by name.
    
    Args:
        collection_name: Name of the collection ("CC18", "regression_2025", etc.)
        study_id: OpenML study ID (optional, will use default for known collections)
        
    Returns:
        List of OpenML task objects
    """
    collection_name = collection_name.lower()
    
    if collection_name in ["cc18", "openml_cc18"]:
        return get_openml_cc18_tasks()
    elif collection_name in ["regression_2025", "openml_regression_2025", "new_openml_suite_2025_regression"]:
        return get_openml_regression_2025_tasks()
    elif study_id is not None:
        return get_openml_tasks_by_study_id(study_id, collection_name)
    else:
        raise ValueError(f"Unknown collection '{collection_name}' and no study_id provided")


def get_openml_tasks_by_study_id(study_id: int, collection_name: str = "custom") -> List[Any]:
    """
    Get tasks from an OpenML study by ID.
    
    Args:
        study_id: OpenML study ID
        collection_name: Name of the collection (for logging)
        
    Returns:
        List of OpenML task objects
    """
    logger.info(f"Fetching OpenML {collection_name} collection (study_id={study_id})")
    
    try:
        # Try the newer API method first
        logger.info(f"Attempting to fetch {collection_name} using newer API method (get_suite)")
        suite = openml.study.get_suite(study_id)
        task_ids = suite.tasks
    except Exception as e1:
        logger.warning(f"Error using get_suite: {e1}")
        try:
            # Try fallback method
            logger.info(f"Attempting fallback method with get_study and entity_type='task'")
            study = openml.study.functions._get_study(study_id, entity_type='task')
            task_ids = study.tasks
        except Exception as e2:
            logger.warning(f"Error using get_study fallback: {e2}")
            logger.error(f"Could not fetch {collection_name} collection with study_id={study_id}")
            return []

    logger.info(f"Retrieved {len(task_ids)} task IDs from {collection_name} collection")
    
    tasks = []
    for task_id in task_ids:
        try:
            task = openml.tasks.get_task(task_id)
            tasks.append(task)
            logger.info(f"Retrieved task {task_id}: {task.get_dataset().name}")
        except Exception as e:
            logger.error(f"Error retrieving task {task_id}: {e}")
    
    logger.info(f"Successfully retrieved {len(tasks)} {collection_name} tasks")
    return tasks


def filter_tasks_by_type(tasks: List[Any], task_type: str) -> List[Any]:
    """
    Filter tasks by type (classification or regression).
    
    Args:
        tasks: List of OpenML task objects
        task_type: Task type to filter for ("classification" or "regression")
        
    Returns:
        Filtered list of OpenML task objects
    """
    filtered_tasks = []
    
    for task in tasks:
        try:
            if task_type.lower() == "classification":
                # Accept both classification types
                if "classification" in task.task_type.lower():
                    filtered_tasks.append(task)
                else:
                    logger.debug(f"Skipping non-classification task {task.task_id} (type: {task.task_type})")
            elif task_type.lower() == "regression":
                if "regression" in task.task_type.lower():
                    filtered_tasks.append(task)
                else:
                    logger.debug(f"Skipping non-regression task {task.task_id} (type: {task.task_type})")
            else:
                # If task_type is not specified, include all tasks
                filtered_tasks.append(task)
        except Exception as e:
            logger.warning(f"Error checking task type for task {task.task_id}: {e}")
    
    logger.info(f"Filtered {len(tasks)} tasks to {len(filtered_tasks)} {task_type} tasks")
    return filtered_tasks