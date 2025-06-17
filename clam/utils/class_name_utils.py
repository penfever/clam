"""
Unified class name handling utilities for CLAM examples.

This module provides consistent class name extraction, normalization, and semantic 
handling across different modalities (audio, vision, tabular) for all examples.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple

from .vlm_prompting import validate_and_clean_class_names

logger = logging.getLogger(__name__)


class ClassNameExtractor:
    """
    Unified class name extractor for all modalities.
    
    Handles:
    - Semantic class name extraction from various sources
    - Normalization to Class_<NUM> format
    - Fallback strategies when semantic names are unavailable
    - Integration with existing validation logic
    """
    
    def __init__(self, semantic_data_dir: Optional[str] = None):
        """
        Initialize the class name extractor.
        
        Args:
            semantic_data_dir: Directory containing semantic data files (e.g., data/cc18_semantic)
        """
        self.semantic_data_dir = semantic_data_dir
        
        # Built-in semantic mappings for common datasets
        self.builtin_mappings = {
            # CIFAR-10 classes
            'cifar10': [
                'airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck'
            ],
            # CIFAR-100 superclasses (simplified)
            'cifar100_coarse': [
                'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_vegetables',
                'household_electrical', 'household_furniture', 'insects', 'large_carnivores',
                'large_man_objects', 'large_natural_scenes', 'large_omnivores', 'medium_mammals',
                'non_insect_invertebrates', 'people', 'reptiles', 'small_mammals',
                'trees', 'vehicles_1', 'vehicles_2'
            ],
            # Common audio emotion classes
            'ravdess_emotions': [
                'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'
            ],
            # ESC-50 categories (sample)
            'esc50_categories': [
                'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow',
                'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops',
                'wind', 'pouring_water', 'toilet_flush', 'thunderstorm',
                'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing', 'footsteps',
                'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping',
                'door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks',
                'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm',
                'clock_tick', 'glass_breaking', 'helicopter', 'chainsaw', 'siren',
                'car_horn', 'engine', 'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw'
            ]
        }
    
    def extract_class_names(
        self,
        labels: Union[List[int], List[str]],
        dataset_name: Optional[str] = None,
        semantic_file: Optional[str] = None,
        use_semantic: bool = True
    ) -> Tuple[List[str], bool]:
        """
        Extract class names from labels with semantic support.
        
        Args:
            labels: List of numeric labels or string class names
            dataset_name: Name of dataset for builtin mappings
            semantic_file: Path to semantic JSON file
            use_semantic: Whether to attempt semantic name extraction
            
        Returns:
            Tuple of (class_names, is_semantic) where is_semantic indicates
            whether semantic names were successfully used
        """
        unique_labels = sorted(set(labels))
        num_classes = len(unique_labels)
        
        if not use_semantic:
            # Direct fallback to Class_<NUM>
            return self._generate_class_num_names(num_classes), False
        
        semantic_names = None
        
        # Try different semantic sources in order of preference
        
        # 1. Try semantic file if provided
        if semantic_file:
            semantic_names = self._load_from_semantic_file(semantic_file, num_classes)
        
        # 2. Try builtin mappings for known datasets
        if semantic_names is None and dataset_name:
            semantic_names = self._get_builtin_mapping(dataset_name, num_classes)
        
        # 3. Try semantic directory if available
        if semantic_names is None and self.semantic_data_dir and dataset_name:
            semantic_names = self._load_from_semantic_dir(dataset_name, num_classes)
        
        # 4. If labels are already strings, try to use them directly
        if semantic_names is None and all(isinstance(label, str) for label in unique_labels):
            try:
                semantic_names = validate_and_clean_class_names([str(label) for label in unique_labels])
                logger.info(f"Using provided string labels as semantic names")
            except ValueError as e:
                logger.warning(f"Failed to validate provided string labels: {e}")
        
        # Apply validation if we have semantic names
        if semantic_names:
            try:
                validated_names = validate_and_clean_class_names(semantic_names)
                if len(validated_names) == num_classes:
                    logger.info(f"Successfully extracted {num_classes} semantic class names")
                    return validated_names, True
                else:
                    logger.warning(f"Semantic names count mismatch: got {len(validated_names)}, expected {num_classes}")
            except ValueError as e:
                logger.warning(f"Semantic name validation failed: {e}")
        
        # Fallback to Class_<NUM> format
        logger.info(f"Using fallback Class_<NUM> format for {num_classes} classes")
        return self._generate_class_num_names(num_classes), False
    
    def normalize_to_class_num(self, num_classes: int) -> List[str]:
        """
        Generate standardized Class_<NUM> names.
        
        Args:
            num_classes: Number of classes
            
        Returns:
            List of class names in Class_0, Class_1, ... format
        """
        return self._generate_class_num_names(num_classes)
    
    def get_semantic_or_fallback(
        self,
        labels: Union[List[int], List[str]],
        dataset_name: Optional[str] = None,
        semantic_file: Optional[str] = None
    ) -> List[str]:
        """
        Convenience method to get semantic names with automatic fallback.
        
        Args:
            labels: List of labels
            dataset_name: Optional dataset name for builtin mappings
            semantic_file: Optional semantic file path
            
        Returns:
            List of class names (semantic if available, otherwise Class_<NUM>)
        """
        class_names, _ = self.extract_class_names(
            labels=labels,
            dataset_name=dataset_name,
            semantic_file=semantic_file,
            use_semantic=True
        )
        return class_names
    
    def _generate_class_num_names(self, num_classes: int) -> List[str]:
        """Generate Class_0, Class_1, ... Class_N names."""
        return [f"Class_{i}" for i in range(num_classes)]
    
    def _get_builtin_mapping(self, dataset_name: str, num_classes: int) -> Optional[List[str]]:
        """Get builtin semantic mapping for known datasets."""
        dataset_key = dataset_name.lower()
        
        # Direct mapping
        if dataset_key in self.builtin_mappings:
            mapping = self.builtin_mappings[dataset_key]
            if len(mapping) >= num_classes:
                return mapping[:num_classes]
        
        # Fuzzy matching for common patterns
        for key, mapping in self.builtin_mappings.items():
            if key in dataset_key or dataset_key in key:
                if len(mapping) >= num_classes:
                    logger.info(f"Using builtin mapping '{key}' for dataset '{dataset_name}'")
                    return mapping[:num_classes]
        
        return None
    
    def _load_from_semantic_file(self, semantic_file: str, num_classes: int) -> Optional[List[str]]:
        """Load semantic class names from a JSON file."""
        try:
            with open(semantic_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            class_names = self._extract_classes_from_semantic_data(data)
            if class_names and len(class_names) >= num_classes:
                logger.info(f"Loaded semantic class names from {semantic_file}")
                return class_names[:num_classes]
            
        except Exception as e:
            logger.warning(f"Failed to load semantic file {semantic_file}: {e}")
        
        return None
    
    def _load_from_semantic_dir(self, dataset_name: str, num_classes: int) -> Optional[List[str]]:
        """Load semantic class names from semantic data directory."""
        if not self.semantic_data_dir or not os.path.exists(self.semantic_data_dir):
            return None
        
        # Try different possible filename patterns
        possible_files = [
            f"{dataset_name}.json",
            f"{dataset_name}_semantic.json",
            f"semantic_{dataset_name}.json"
        ]
        
        for filename in possible_files:
            filepath = os.path.join(self.semantic_data_dir, filename)
            if os.path.exists(filepath):
                return self._load_from_semantic_file(filepath, num_classes)
        
        return None
    
    def _extract_classes_from_semantic_data(self, data: Dict[str, Any]) -> Optional[List[str]]:
        """Extract class names from semantic data structure."""
        
        # Try different common structures
        
        # Structure 1: Direct class_names array
        if 'class_names' in data and isinstance(data['class_names'], list):
            return data['class_names']
        
        # Structure 2: target_values dictionary
        if 'target_values' in data:
            target_values = data['target_values']
            if isinstance(target_values, dict):
                return sorted(target_values.keys())
            elif isinstance(target_values, list):
                return [str(val) for val in target_values]
        
        # Structure 3: target_variable with values
        if 'target_variable' in data:
            target_var = data['target_variable']
            if isinstance(target_var, dict) and 'values' in target_var:
                values = target_var['values']
                if isinstance(values, list):
                    return [str(val) for val in values]
                elif isinstance(values, dict):
                    return sorted(values.keys())
        
        # Structure 4: classes array
        if 'classes' in data and isinstance(data['classes'], list):
            return data['classes']
        
        # Structure 5: labels dictionary/array
        if 'labels' in data:
            labels = data['labels']
            if isinstance(labels, dict):
                return sorted(labels.keys())
            elif isinstance(labels, list):
                return [str(label) for label in labels]
        
        return None


# Convenience functions for backward compatibility and ease of use

def extract_class_names_from_labels(
    labels: Union[List[int], List[str]],
    dataset_name: Optional[str] = None,
    semantic_file: Optional[str] = None,
    semantic_data_dir: Optional[str] = None,
    use_semantic: bool = True
) -> Tuple[List[str], bool]:
    """
    Convenience function to extract class names from labels.
    
    Args:
        labels: List of numeric labels or string class names
        dataset_name: Name of dataset for builtin mappings
        semantic_file: Path to semantic JSON file
        semantic_data_dir: Directory containing semantic data files
        use_semantic: Whether to attempt semantic name extraction
        
    Returns:
        Tuple of (class_names, is_semantic)
    """
    extractor = ClassNameExtractor(semantic_data_dir=semantic_data_dir)
    return extractor.extract_class_names(
        labels=labels,
        dataset_name=dataset_name,
        semantic_file=semantic_file,
        use_semantic=use_semantic
    )


def normalize_class_names_to_class_num(num_classes: int) -> List[str]:
    """
    Generate standardized Class_<NUM> names.
    
    Args:
        num_classes: Number of classes
        
    Returns:
        List of class names in Class_0, Class_1, ... format
    """
    return [f"Class_{i}" for i in range(num_classes)]


def get_semantic_class_names_or_fallback(
    labels: Union[List[int], List[str]],
    dataset_name: Optional[str] = None,
    semantic_file: Optional[str] = None,
    semantic_data_dir: Optional[str] = None
) -> List[str]:
    """
    Get semantic class names with automatic fallback to Class_<NUM> format.
    
    Args:
        labels: List of labels
        dataset_name: Optional dataset name for builtin mappings
        semantic_file: Optional semantic file path
        semantic_data_dir: Directory containing semantic data files
        
    Returns:
        List of class names (semantic if available, otherwise Class_<NUM>)
    """
    extractor = ClassNameExtractor(semantic_data_dir=semantic_data_dir)
    return extractor.get_semantic_or_fallback(
        labels=labels,
        dataset_name=dataset_name,
        semantic_file=semantic_file
    )