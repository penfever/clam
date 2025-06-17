"""
Unified VLM prompting utilities for LLATA.

This module provides consistent prompting strategies across different modalities
(tabular, audio, image) for Vision Language Model classification tasks.
"""

import logging
import re
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


def validate_and_clean_class_names(class_names: List[str]) -> List[str]:
    """
    Validate and clean class names for semantic naming.
    
    Requirements:
    1. Unique names
    2. Only ASCII characters  
    3. Less than 30 characters per name
    4. No whitespace (replace with underscores)
    
    Args:
        class_names: List of class names to validate
        
    Returns:
        List of cleaned and validated class names
        
    Raises:
        ValueError: If validation fails
    """
    if not class_names:
        return class_names
        
    cleaned_names = []
    seen_names = set()
    
    for i, name in enumerate(class_names):
        # Convert to string if not already
        name_str = str(name)
        
        # Replace whitespace with underscores and remove/replace special characters
        cleaned_name = re.sub(r'\s+', '_', name_str)
        # Replace common special characters with underscores
        cleaned_name = re.sub(r'[/\\|,.;:!@#$%^&*()+=\[\]{}"`~<>?]', '_', cleaned_name)
        # Remove multiple consecutive underscores
        cleaned_name = re.sub(r'_+', '_', cleaned_name)
        # Remove leading/trailing underscores
        cleaned_name = cleaned_name.strip('_')
        
        # Check ASCII only
        if not cleaned_name.isascii():
            raise ValueError(f"Class name at index {i} contains non-ASCII characters: '{name_str}' -> '{cleaned_name}'")
        
        # Check length and truncate if necessary
        if len(cleaned_name) > 30:
            original_name = cleaned_name
            cleaned_name = cleaned_name[:27] + "..."
            logger.warning(f"Class name at index {i} too long ({len(original_name)} chars), truncated: '{original_name}' -> '{cleaned_name}'")
        
        # Ensure not empty after cleaning
        if not cleaned_name or cleaned_name == '_':
            cleaned_name = f"class_{i}"
            logger.warning(f"Empty class name at index {i}, using fallback: '{cleaned_name}'")
        
        # Check uniqueness
        if cleaned_name in seen_names:
            # Make unique by appending counter, ensuring we stay under 30 chars
            original_cleaned = cleaned_name
            counter = 1
            while cleaned_name in seen_names:
                suffix = f"_{counter}"
                if len(original_cleaned) + len(suffix) > 30:
                    # Truncate base name to fit suffix
                    base_name = original_cleaned[:30 - len(suffix)]
                    cleaned_name = f"{base_name}{suffix}"
                else:
                    cleaned_name = f"{original_cleaned}{suffix}"
                counter += 1
            logger.warning(f"Duplicate class name '{original_cleaned}' at index {i}, using '{cleaned_name}'")
        
        seen_names.add(cleaned_name)
        cleaned_names.append(cleaned_name)
    
    return cleaned_names


def create_classification_prompt(
    class_names: List[str],
    modality: str = "tabular",
    use_knn: bool = False,
    use_3d: bool = False,
    knn_k: Optional[int] = None,
    legend_text: Optional[str] = None,
    include_spectrogram: bool = False,
    dataset_description: Optional[str] = None,
    use_semantic_names: bool = False
) -> str:
    """
    Create a classification prompt for VLM based on modality and visualization type.
    
    Args:
        class_names: List of class names/labels
        modality: Type of data ("tabular", "audio", "image")
        use_knn: Whether KNN connections are shown
        use_3d: Whether 3D visualization is used
        knn_k: Number of nearest neighbors (if use_knn=True)
        legend_text: Legend text from the visualization
        include_spectrogram: Whether spectrogram is included (for audio)
        dataset_description: Optional description of the dataset/task
        use_semantic_names: Whether to use semantic class names in prompts (default: False uses "Class X")
        
    Returns:
        Formatted prompt string
    """
    # Format class list consistently
    if use_semantic_names and not all(isinstance(name, (int, float)) for name in class_names):
        # Validate and clean semantic names
        try:
            cleaned_class_names = validate_and_clean_class_names(class_names)
            class_list_str = ", ".join([f'"{name}"' for name in cleaned_class_names])
            class_format_example = f'"{cleaned_class_names[0]}", "{cleaned_class_names[1]}", etc.'
        except ValueError as e:
            logger.warning(f"Class name validation failed: {e}. Falling back to 'Class X' format.")
            # Fall back to Class X format
            class_indices = list(range(len(class_names)))
            class_list_str = ", ".join([f'"Class {i}"' for i in class_indices])
            class_format_example = '"Class 0", "Class 1", "Class 2"'
            use_semantic_names = False  # Override to ensure consistent parsing
    else:
        # Default: Always use "Class X" format for consistency with legends
        class_indices = list(range(len(class_names)))
        class_list_str = ", ".join([f'"Class {i}"' for i in class_indices])
        class_format_example = '"Class 0", "Class 1", "Class 2"'
    
    # Create modality-specific description
    if modality == "audio":
        data_description = f"""Looking at this{'enhanced' if use_knn else ''} {'3D ' if use_3d else ''}t-SNE visualization of audio classification data, you can see:

1. Colored points representing training audio samples, where each color corresponds to a different class
2. {'Gray square points representing test audio samples' if not use_knn else 'Test points (if any) shown as gray squares'}
3. One red star point which is the query audio sample I want you to classify"""
        
        if use_3d:
            data_description += "\n4. Four different views of the same 3D space: Isometric, Front (XZ), Side (YZ), and Top (XY)"
            
        if use_knn and knn_k:
            data_description += f"\n{5 if use_3d else 4}. A pie chart showing the distribution of the {knn_k} nearest neighbors by class"
            data_description += f"\n{6 if use_3d else 5}. The pie chart includes class counts, percentages, and average distances to neighbors"
            
        if include_spectrogram:
            data_description += f"\n{7 if use_knn and use_3d else 6 if use_knn or use_3d else 5}. Audio spectrogram of the query sample shown below the t-SNE plot"
            
    elif modality == "tabular":
        data_description = f"""Looking at this{'enhanced' if use_knn else ''} {'3D ' if use_3d else ''}t-SNE visualization of tabular data, you can see:

1. Colored points representing training data, where each color corresponds to a different class
2. Gray square points representing test data  
3. One red star point which is the query point I want you to classify"""
        
        if use_3d:
            data_description += "\n4. Four different views of the same 3D space: Isometric, Front (XZ), Side (YZ), and Top (XY)"
            
        if use_knn and knn_k:
            data_description += f"\n{5 if use_3d else 4}. A pie chart showing the distribution of the {knn_k} nearest neighbors by class"
            data_description += f"\n{6 if use_3d else 5}. The pie chart includes class counts, percentages, and average distances to neighbors"
            
    else:  # image or other
        data_description = f"""Looking at this{'enhanced' if use_knn else ''} {'3D ' if use_3d else ''}t-SNE visualization of {modality} data, you can see:

1. Colored points representing training samples, where each color corresponds to a different class
2. Gray square points representing test samples  
3. One red star point which is the query sample I want you to classify"""
        
        if use_3d:
            data_description += "\n4. Four different views of the same 3D space: Isometric, Front (XZ), Side (YZ), and Top (XY)"
            
        if use_knn and knn_k:
            data_description += f"\n{5 if use_3d else 4}. A pie chart showing the distribution of the {knn_k} nearest neighbors by class"
            data_description += f"\n{6 if use_3d else 5}. The pie chart includes class counts, percentages, and average distances to neighbors"

    # Add legend text if provided
    if legend_text:
        data_description += f"\n\n{legend_text}"

    # Add dataset description if provided
    dataset_context = ""
    if dataset_description:
        dataset_context = f"\n\nDataset Context: {dataset_description}"

    # Create modality-specific important notes
    if use_knn:
        important_note = f"\nIMPORTANT: The pie chart shows the class distribution of the {knn_k} nearest neighbors found in the original {'Whisper ' if modality == 'audio' else ''}{'high-dimensional ' if modality == 'tabular' else ''}embedding space, NOT just based on the {'3D' if use_3d else '2D'} visualization space. Smaller average distances indicate higher similarity."
    else:
        important_note = ""

    # Create analysis instructions
    if use_knn:
        analysis_prompt = f"Based on BOTH the spatial position in the t-SNE visualization AND the explicit nearest neighbor connections, which class should this query {'audio sample' if modality == 'audio' else 'point'} belong to? The available classes are: {class_list_str}"
        
        considerations = ["The spatial clustering patterns" + (" across all four 3D views" if use_3d else " in the t-SNE visualization")]
        considerations.append("Which classes the nearest neighbors (connected by red lines) belong to")
        considerations.append("The relative importance of close neighbors (thicker lines)")
        
        if include_spectrogram and modality == "audio":
            considerations.append("The audio spectrogram patterns that might provide additional context")
    else:
        analysis_prompt = f"Based on the position of the red star (query {'audio sample' if modality == 'audio' else 'point'}) relative to the colored training points{' across ALL viewing angles' if use_3d else ''}, which class should this query {'audio sample' if modality == 'audio' else 'point'} belong to? The available classes are: {class_list_str}"
        
        considerations = [f"The spatial relationships in {'3D space by examining all four views' if use_3d else 'the t-SNE visualization'}"]
        considerations.append("Which colored class clusters the red star is closest to or embedded within")
        
        if include_spectrogram and modality == "audio":
            considerations.append("The audio spectrogram patterns that might provide additional context")

    consider_text = "\n".join([f"- {consideration}" for consideration in considerations])

    # Create response format instruction
    if use_knn:
        analysis_type = "spatial clustering AND the pie chart neighbor analysis"
    elif use_3d:
        analysis_type = "3D spatial clustering patterns you observe across the multiple views"
    else:
        analysis_type = "spatial clustering patterns you observe"
    
    spectrogram_text = " and spectrogram analysis" if include_spectrogram and modality == "audio" else ""
    response_format = f'Please respond with just the class label (e.g., {class_format_example}) followed by a brief explanation of your reasoning based on the {analysis_type}{spectrogram_text}.'

    # Combine all parts
    prompt = f"""{data_description}{dataset_context}{important_note}

{analysis_prompt}

Consider:
{consider_text}

{response_format}

Format your response as: "Class: [class_label] | Reasoning: [brief explanation]" """

    return prompt


def parse_vlm_response(response: str, unique_classes: List, logger_instance: Optional[logging.Logger] = None, use_semantic_names: bool = False) -> Any:
    """
    Parse VLM response to extract the predicted class.
    
    Args:
        response: Raw VLM response string
        unique_classes: List of valid class labels
        logger_instance: Logger for debugging
        use_semantic_names: Whether semantic names were used in the prompt
        
    Returns:
        Predicted class (same type as unique_classes elements)
    """
    if logger_instance is None:
        logger_instance = logger
        
    response_lower = response.lower().strip()
    
    # Try to parse structured response format first
    if "class:" in response_lower:
        try:
            # Extract text after "class:"
            class_part = response.split(":", 1)[1].split("|")[0].strip()
            
            # Remove quotes if present
            class_part = class_part.strip('"\'')
            
            # If using "Class X" format, extract the number
            if not use_semantic_names and class_part.lower().startswith("class "):
                try:
                    class_num_str = class_part.lower().replace("class ", "").strip()
                    class_num = int(class_num_str)
                    if 0 <= class_num < len(unique_classes):
                        logger_instance.debug(f"Parsed Class {class_num} -> {unique_classes[class_num]}")
                        return unique_classes[class_num]
                except (ValueError, IndexError):
                    pass
            
            # Try to match with available classes (semantic names or direct)
            if use_semantic_names:
                for cls in unique_classes:
                    if str(cls).lower() == class_part.lower():
                        logger_instance.debug(f"Parsed structured response: '{class_part}' -> {cls}")
                        return cls
                    
        except Exception as e:
            logger_instance.warning(f"Error parsing structured response: {e}")
    
    # Fallback: Look for "Class X" pattern anywhere in response
    if not use_semantic_names:
        import re
        class_pattern = r'\bclass\s+(\d+)\b'
        match = re.search(class_pattern, response_lower)
        if match:
            try:
                class_num = int(match.group(1))
                if 0 <= class_num < len(unique_classes):
                    logger_instance.debug(f"Found Class {class_num} pattern -> {unique_classes[class_num]}")
                    return unique_classes[class_num]
            except (ValueError, IndexError):
                pass
    
    # Fallback: Look for any class name in the response (for semantic names)
    if use_semantic_names:
        for cls in unique_classes:
            cls_str = str(cls).lower()
            if cls_str in response_lower:
                # Check if it's a whole word match (not part of another word)
                import re
                if re.search(r'\b' + re.escape(cls_str) + r'\b', response_lower):
                    logger_instance.debug(f"Found class in response: '{cls_str}' -> {cls}")
                    return cls
    
    # Final fallback: Return first class
    logger_instance.warning(f"Could not parse class from response: '{response}'. Using fallback: {unique_classes[0]}")
    return unique_classes[0]


def create_vlm_conversation(image, prompt: str) -> List[Dict]:
    """
    Create a conversation structure for VLM input.
    
    Args:
        image: PIL Image object
        prompt: Text prompt string
        
    Returns:
        Conversation structure for VLM
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