"""
VLM (Vision Language Model) utilities for LLATA.

This module provides common functionality for working with VLMs across
different LLATA implementations, including response parsing, conversation
formatting, and error handling.
"""

import re
import logging
from typing import Any, List, Dict, Union, Optional
import numpy as np

logger = logging.getLogger(__name__)


def parse_vlm_response(response: str, unique_classes: np.ndarray, logger_instance: Optional[logging.Logger] = None) -> Any:
    """
    Parse VLM response to extract predicted class.
    
    This function implements a robust parsing strategy that works across
    different VLM response formats and class types.
    
    Args:
        response: Raw VLM response string
        unique_classes: Array of valid class labels
        logger_instance: Logger instance for warnings (optional)
        
    Returns:
        Parsed class label from unique_classes (as Python native type)
    """
    if logger_instance is None:
        logger_instance = logger
        
    response = response.strip()
    
    # Helper function to convert numpy types to Python native types
    def to_native_type(value):
        if hasattr(value, 'item'):  # numpy scalar
            return value.item()
        return value
    
    # Try to find "Class: X" pattern (most common format)
    class_match = re.search(r'Class:\s*([^\s|]+)', response, re.IGNORECASE)
    if class_match:
        class_str = class_match.group(1).strip()
        
        # Try to convert to appropriate type and match with unique_classes
        for cls in unique_classes:
            try:
                if str(cls) == class_str or cls == class_str:
                    return to_native_type(cls)
                # Try numeric conversion
                if isinstance(cls, (int, float)) or hasattr(cls, 'item'):
                    if float(cls) == float(class_str):
                        return to_native_type(cls)
            except (ValueError, TypeError):
                continue
    
    # Fallback: look for any mention of class labels in the response
    for cls in unique_classes:
        if str(cls) in response:
            return to_native_type(cls)
    
    # Final fallback: return first class
    logger_instance.warning(f"Could not parse class from VLM response: '{response[:100]}...'. Using fallback.")
    fallback_class = unique_classes[0]
    
    # Convert numpy types to Python native types
    if hasattr(fallback_class, 'item'):  # numpy scalar
        return fallback_class.item()
    else:
        return fallback_class


def create_classification_prompt(
    legend_text: str,
    class_list_str: str,
    use_3d: bool = False,
    use_knn: bool = False,
    knn_k: int = 5,
    backend_name: str = "t-SNE",
    domain: str = "tabular"
) -> str:
    """
    Create standardized classification prompt for VLMs.
    
    Args:
        legend_text: Legend describing the visualization
        class_list_str: Comma-separated string of available classes
        use_3d: Whether using 3D visualization
        use_knn: Whether using KNN connections
        knn_k: Number of nearest neighbors (if use_knn=True)
        backend_name: Name of dimensionality reduction backend (t-SNE, PCA, etc.)
        domain: Data domain (tabular, image, etc.)
        
    Returns:
        Formatted prompt string
    """
    if use_knn:
        if use_3d:
            prompt = f"""Looking at this enhanced 3D {backend_name} visualization of {domain} data shown from multiple viewing angles:

1. Colored points represent training data, where each color corresponds to a different class
2. Gray square points represent test data  
3. One red star point which is the query point I want you to classify
4. A pie chart showing the class distribution of the {knn_k} nearest neighbors in the original embedding space
5. Four different views of the same 3D space: Isometric, Front (XZ), Side (YZ), and Top (XY)

{legend_text}

IMPORTANT: The pie chart shows which training point classes are most similar to the query point in the original feature space. Pie slice sizes show the count of neighbors from each class, and slice prominence (explosion) indicates how close those neighbors are on average.

Based on BOTH the spatial position across all 3D viewing angles AND the K-nearest neighbor pie chart, which class should this query point belong to? The available classes are: {class_list_str}

Consider:
- The spatial clustering patterns across all four 3D views
- Which classes dominate the K-NN pie chart (larger slices)
- Which classes have the closest neighbors (more prominent/exploded slices)
- The balance between spatial clustering and embedding similarity

Please respond with just the class label (e.g., "0", "1", "2", etc.) followed by a brief explanation that references both the spatial clustering AND the K-NN pie chart information.

Format your response as: "Class: [class_label] | Reasoning: [brief explanation]" """
        else:
            prompt = f"""Looking at this enhanced {backend_name} visualization of {domain} data:

1. Colored points represent training data, where each color corresponds to a different class
2. Gray square points represent test data  
3. One red star point which is the query point I want you to classify
4. A pie chart showing the class distribution of the {knn_k} nearest neighbors in the original embedding space

{legend_text}

IMPORTANT: The pie chart shows which training point classes are most similar to the query point in the original feature space. Pie slice sizes show the count of neighbors from each class, and slice prominence (explosion) indicates how close those neighbors are on average.

Based on BOTH the spatial position in the {backend_name} visualization AND the K-nearest neighbor pie chart, which class should this query point belong to? The available classes are: {class_list_str}

Consider:
- The spatial clustering patterns in the {backend_name} visualization
- Which classes dominate the K-NN pie chart (larger slices)
- Which classes have the closest neighbors (more prominent/exploded slices)
- The balance between spatial clustering and embedding similarity

Please respond with just the class label (e.g., "0", "1", "2", etc.) followed by a brief explanation that references both the spatial clustering AND the K-NN pie chart information.

Format your response as: "Class: [class_label] | Reasoning: [brief explanation]" """
    else:
        if use_3d:
            prompt = f"""Looking at this 3D {backend_name} visualization of {domain} data shown from multiple viewing angles:

1. Colored points represent training data, where each color corresponds to a different class
2. Gray square points represent test data  
3. One red star point which is the query point I want you to classify
4. Four different views of the same 3D space: Isometric, Front (XZ), Side (YZ), and Top (XY)

{legend_text}

Based on the position of the red star (query point) relative to the colored training points across ALL viewing angles, which class should this query point belong to? The available classes are: {class_list_str}

Consider the spatial relationships in 3D space by examining all four views. Look for which colored class clusters the red star is closest to or embedded within.

Please respond with just the class label (e.g., "0", "1", "2", etc.) followed by a brief explanation of your reasoning based on the 3D spatial clustering patterns you observe across the multiple views.

Format your response as: "Class: [class_label] | Reasoning: [brief explanation]" """
        else:
            prompt = f"""Looking at this {backend_name} visualization of {domain} data:

1. Colored points represent training data, where each color corresponds to a different class
2. Gray square points represent test data  
3. One red star point which is the query point I want you to classify

{legend_text}

Based on the position of the red star (query point) relative to the colored training points, which class should this query point belong to? The available classes are: {class_list_str}

Please respond with just the class label (e.g., "0", "1", "2", etc.) followed by a brief explanation of your reasoning based on the spatial clustering patterns you observe.

Format your response as: "Class: [class_label] | Reasoning: [brief explanation]" """
    
    return prompt


def create_vlm_conversation(image, prompt: str) -> List[Dict]:
    """
    Create standardized VLM conversation format.
    
    Args:
        image: PIL Image object
        prompt: Text prompt for the VLM
        
    Returns:
        Conversation in standard format for VLM processing
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]