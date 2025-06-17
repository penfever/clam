#!/usr/bin/env python
"""
TabLLM baseline evaluation module.
Contains functions for evaluating the TabLLM baseline on tabular datasets.
"""

import os
import sys
import numpy as np
import torch
import json
import glob
import time
import logging
import math
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional, List, Tuple
from clam.utils.model_loader import model_loader, GenerationConfig

def load_tabllm_config_by_openml_id(openml_task_id, original_feature_count=None):
    """Load TabLLM configuration by OpenML task ID with feature count validation.
    
    Args:
        openml_task_id: OpenML task ID to look up
        original_feature_count: Number of features in dataset before preprocessing
        
    Returns:
        Tuple of (template_data, feature_mapping) or (None, None)
    """
    logger = logging.getLogger(__name__)
    
    # Use relative path from current script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tabllm_dir = os.path.join(current_dir, "tabllm_like")
    
    # Load OpenML task mapping
    mapping_path = os.path.join(tabllm_dir, "openml_task_mapping.json")
    if not os.path.exists(mapping_path):
        logger.warning(f"OpenML task mapping not found at {mapping_path}")
        return None, None
    
    try:
        with open(mapping_path, 'r') as f:
            task_mapping = json.load(f)
    except Exception as e:
        logger.error(f"Error loading OpenML task mapping: {e}")
        return None, None
    
    # Find dataset name that maps to this OpenML task ID
    dataset_name = None
    for name, task_id in task_mapping.items():
        if task_id == openml_task_id:
            dataset_name = name
            break
    
    if dataset_name is None:
        logger.debug(f"No TabLLM config found for OpenML task ID {openml_task_id}")
        return None, None
    
    # Load the corresponding template file
    template_path = os.path.join(tabllm_dir, f"templates_{dataset_name}.yaml")
    
    try:
        if os.path.exists(template_path):
            import yaml
            
            # Define constructors for custom YAML tags
            def template_constructor(loader, node):
                return loader.construct_mapping(node)
            def template_metadata_constructor(loader, node):
                return loader.construct_mapping(node)
            
            # Create a custom loader
            CustomLoader = yaml.SafeLoader
            CustomLoader.add_constructor('!Template', template_constructor)
            CustomLoader.add_constructor('!TemplateMetadata', template_metadata_constructor)
            
            with open(template_path, 'r') as f:
                template_data = yaml.load(f, Loader=CustomLoader)
            
            logger.info(f"Found TabLLM config for OpenML task {openml_task_id} (dataset: {dataset_name})")
            
            # Load semantic information for feature count validation
            semantic_dir = "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/clam/clam/data/cc18_semantic_complete"
            semantic_file = os.path.join(semantic_dir, f"{openml_task_id}.json")
            
            if os.path.exists(semantic_file) and original_feature_count is not None:
                try:
                    with open(semantic_file, 'r') as f:
                        semantic_info = json.load(f)
                    
                    # Count features from semantic info
                    config_feature_count = None
                    if 'columns' in semantic_info:
                        config_feature_count = len([col for col in semantic_info['columns'] if col.get('name') != 'target'])
                    elif 'feature_descriptions' in semantic_info:
                        config_feature_count = len(semantic_info['feature_descriptions'])
                    elif 'feature_description' in semantic_info:
                        config_feature_count = len(semantic_info['feature_description'])
                    
                    if config_feature_count is not None and config_feature_count != original_feature_count:
                        error_msg = (
                            f"Feature count mismatch for OpenML task {openml_task_id}: "
                            f"dataset has {original_feature_count} features but TabLLM config expects {config_feature_count} features. "
                            f"This indicates a version mismatch or preprocessing differences."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    logger.info(f"Feature count validation passed: {original_feature_count} features")
                    
                    # Create feature mapping to preserve semantic descriptions
                    feature_mapping = {
                        'semantic_info': semantic_info,
                        'task_id': openml_task_id,
                        'dataset_name': dataset_name
                    }
                    
                except Exception as e:
                    logger.warning(f"Could not validate feature count from semantic info: {e}")
                    feature_mapping = None
            else:
                feature_mapping = None
            
            return template_data, feature_mapping
        
        logger.debug(f"TabLLM template file not found: {template_path}")
        return None, None
        
    except ValueError:
        # Re-raise validation errors (like feature count mismatch)
        raise
    except Exception as e:
        logger.error(f"Error loading TabLLM config for OpenML task {openml_task_id}: {e}")
        return None, None

def load_tabllm_template(dataset_name):
    """Load TabLLM template if available for the dataset (legacy function)."""
    # Use relative path from current script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, "tabllm_like", f"templates_{dataset_name}.yaml")
    
    try:
        if os.path.exists(template_path):
            import yaml
            
            # Define constructors for custom YAML tags
            def template_constructor(loader, node):
                return loader.construct_mapping(node)
            def template_metadata_constructor(loader, node):
                return loader.construct_mapping(node)
            
            # Create a custom loader
            CustomLoader = yaml.SafeLoader
            CustomLoader.add_constructor('!Template', template_constructor)
            CustomLoader.add_constructor('!TemplateMetadata', template_metadata_constructor)
            
            with open(template_path, 'r') as f:
                template_data = yaml.load(f, Loader=CustomLoader)
            return template_data
        return None
    except Exception as e:
        logging.getLogger(__name__).debug(f"Could not load template for {dataset_name}: {e}")
        return None

def create_feature_mapping_after_preprocessing(original_feature_names, processed_feature_names, feature_mapping):
    """Create mapping from processed feature names to original semantic descriptions.
    
    Args:
        original_feature_names: List of original feature names from dataset
        processed_feature_names: List of feature names after preprocessing
        feature_mapping: Original feature mapping from TabLLM config
        
    Returns:
        Dictionary mapping processed feature names to semantic info
    """
    if feature_mapping is None or 'semantic_info' not in feature_mapping:
        return {}
    
    semantic_info = feature_mapping['semantic_info']
    processed_mapping = {
        'semantic_info': {},
        'task_id': feature_mapping.get('task_id'),
        'dataset_name': feature_mapping.get('dataset_name')
    }
    
    # Extract original descriptions based on semantic info structure
    original_descriptions = {}
    if 'columns' in semantic_info:
        for col in semantic_info['columns']:
            if col.get('name') != 'target':
                original_descriptions[col['name']] = col.get('semantic_description', col['name'])
    elif 'feature_descriptions' in semantic_info:
        original_descriptions = semantic_info['feature_descriptions']
    elif 'feature_description' in semantic_info:
        original_descriptions = semantic_info['feature_description']
    
    # Create mapping from original to processed names
    for i, processed_name in enumerate(processed_feature_names):
        if i < len(original_feature_names):
            original_name = original_feature_names[i]
            if original_name in original_descriptions:
                processed_mapping['semantic_info'][processed_name] = original_descriptions[original_name]
            # Also try to match by name similarity for robustness
            else:
                # Look for partial matches in original descriptions
                for orig_key, description in original_descriptions.items():
                    if orig_key.lower() in processed_name.lower() or processed_name.lower() in orig_key.lower():
                        processed_mapping['semantic_info'][processed_name] = description
                        break
    
    return processed_mapping

def load_tabllm_note_examples(dataset_name):
    """Load TabLLM note examples if available for the dataset."""
    # Use relative path from current script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    note_path = os.path.join(current_dir, "tabllm_like", f"note_{dataset_name}_example.txt")
    
    try:
        if os.path.exists(note_path):
            with open(note_path, 'r') as f:
                content = f.read().strip()
            # Parse the example format: ['The age is 72. The workclass is Self-emp-inc...']
            if content.startswith("['") and content.endswith("']"):
                example_note = content[2:-2]  # Remove [' and ']
                return example_note
        return None
    except Exception as e:
        logging.getLogger(__name__).debug(f"Could not load note example for {dataset_name}: {e}")
        return None

def load_semantic_info(dataset_name, semantic_dir):
    """Load semantic information for a dataset."""
    logger = logging.getLogger(__name__)
    
    try:
        # Look for semantic information for this dataset
        semantic_file = os.path.join(semantic_dir, f"{dataset_name}.json")
        if not os.path.exists(semantic_file):
            # Try alternative names (OpenML ID might be used)
            semantic_files = glob.glob(os.path.join(semantic_dir, "*.json"))
            for sf in semantic_files:
                with open(sf, 'r') as f:
                    semantic_info = json.load(f)
                    if semantic_info.get('dataset_name') == dataset_name or semantic_info.get('dataset') == dataset_name:
                        semantic_file = sf
                        break
        
        if os.path.exists(semantic_file):
            logger.info(f"Loading semantic information for {dataset_name}")
            with open(semantic_file, 'r') as f:
                semantic_info = json.load(f)
            return semantic_info
        
        return None
    
    except Exception as e:
        print(f"Could not load semantic information for {dataset_name}: {e}")
        return None

def generate_tabllm_data_on_demand(dataset_name, semantic_dir):
    """Generate TabLLM data on-demand using the synthesize_tabllm_real_data.py script logic."""
    logger = logging.getLogger(__name__)
    
    try:
        # Import the synthesis functions
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tabllm_like_path = os.path.join(current_dir, "tabllm_like")
        if tabllm_like_path not in sys.path:
            sys.path.append(tabllm_like_path)
        
        from synthesize_tabllm_real_data import generate_note_from_semantic_info, create_template_for_dataset
        
        # Load semantic information
        semantic_info = load_semantic_info(dataset_name, semantic_dir)
        
        if semantic_info:
            logger.info(f"Generating TabLLM data on-demand for {dataset_name}")
            
            # Generate note and template
            note = generate_note_from_semantic_info(semantic_info)
            template_data = create_template_for_dataset(semantic_info)
            
            return note, template_data
        
        return None, None
    
    except Exception as e:
        print(f"Could not generate TabLLM data on-demand for {dataset_name}: {e}")
        return None, None

def evaluate_tabllm(dataset, args):
    """Evaluate TabLLM baseline using proper ICL methodology with log probability computation."""
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating TabLLM on dataset {dataset['name']}")
    logger.info(f"Using num_few_shot_examples={args.num_few_shot_examples}")
    
    # Get original feature count and names before preprocessing
    X, y = dataset["X"], dataset["y"]
    original_feature_count = X.shape[1] if hasattr(X, 'shape') else len(X[0])
    original_feature_names = dataset.get("attribute_names", [])
    
    # Load TabLLM configuration by OpenML ID if available
    openml_task_id = dataset.get('id')
    template_data, feature_mapping = None, None
    
    if openml_task_id:
        template_data, feature_mapping = load_tabllm_config_by_openml_id(
            openml_task_id, 
            original_feature_count
        )
        if template_data:
            logger.info(f"Using TabLLM metadata for OpenML task {openml_task_id} (dataset: {dataset['name']})")
        else:
            logger.info(f"No TabLLM metadata found for OpenML task {openml_task_id} (dataset: {dataset['name']}), using default approach")
    else:
        logger.warning(f"No OpenML task ID found for dataset {dataset['name']}, cannot load TabLLM config")
    
    # Fallback to legacy loading if no OpenML ID config found
    if template_data is None:
        template_data = load_tabllm_template(dataset['name'])
        example_note = load_tabllm_note_examples(dataset['name'])
        
        # Try to load semantic information for better feature descriptions
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels from examples to clam root
        clam_root = os.path.dirname(os.path.dirname(current_dir))
        # Try both cc18_semantic_complete and cc18_semantic directories
        semantic_dirs = [
            os.path.join(clam_root, "data", "cc18_semantic_complete"),
            os.path.join(clam_root, "data", "cc18_semantic")
        ]
        semantic_dir = None
        for dir_path in semantic_dirs:
            if os.path.exists(dir_path):
                semantic_dir = dir_path
                break
        
        if semantic_dir is None:
            logger.warning(f"No semantic directory found. Tried: {semantic_dirs}")
            semantic_dir = semantic_dirs[0]  # Use first as fallback
        semantic_info = load_semantic_info(dataset['name'], semantic_dir)
        
        # If no pre-generated data available, try to generate on-demand
        if template_data is None or example_note is None:
            generated_note, generated_template = generate_tabllm_data_on_demand(dataset['name'], semantic_dir)
            
            if template_data is None and generated_template is not None:
                template_data = generated_template
                logger.info(f"Generated template on-demand for {dataset['name']}")
            
            if example_note is None and generated_note is not None:
                example_note = generated_note
                logger.info(f"Generated note example on-demand for {dataset['name']}")
        
        if template_data is not None:
            logger.info(f"Using TabLLM semantic templates for {dataset['name']}")
        else:
            logger.info(f"No TabLLM templates available for {dataset['name']}, using default approach")
        
        if semantic_info is not None:
            logger.info(f"Using semantic information for enhanced feature descriptions in {dataset['name']}")
        else:
            logger.info(f"No semantic information available for {dataset['name']}, using raw feature names")
    
    # Import required utilities
    from clam.utils import (
        drop_feature_for_oom,
        is_oom_error,
        create_tabllm_note,
        apply_feature_reduction,
        unified_llm_predict
    )
    
    # Import regenerate_few_shot_examples from llm_evaluation_utils (not exported in __init__)
    from clam.utils.llm_evaluation_utils import regenerate_few_shot_examples
    
    # Split data
    X, y = dataset["X"], dataset["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )
    
    # Limit test samples if specified
    if args.max_test_samples and args.max_test_samples < len(X_test):
        X_test = X_test[:args.max_test_samples]
        y_test = y_test[:args.max_test_samples]
    
    # Apply feature selection using unified approach
    X_train, X_test, dataset, _ = apply_feature_reduction(
        X_train, y_train, X_test, dataset, args, logger
    )
    
    # Update feature mapping after preprocessing
    processed_feature_names = dataset.get("attribute_names", [])
    processed_feature_mapping = create_feature_mapping_after_preprocessing(
        original_feature_names, processed_feature_names, feature_mapping
    )
    
    start_time = time.time()
    
    try:
        # Use modern model instead of T0
        model_name = args.tabllm_model if hasattr(args, 'tabllm_model') else "Qwen/QwQ-32B-Preview"
        
        # Load model using centralized model loader with VLLM support
        try:
            # Configure model loading parameters
            model_kwargs = {
                'low_cpu_mem_usage': True,
                'use_cache': False  # Disable KV cache to save memory
            }
            
            # Configure device and dtype
            if torch.cuda.is_available() and args.device != "cpu":
                model_kwargs.update({
                    'torch_dtype': torch.float16,
                    'device_map': "auto" if args.gpu_index == 0 else None
                })
            else:
                model_kwargs.update({
                    'torch_dtype': torch.float32
                })
            
            # For VLLM, add tensor parallel size if using multiple GPUs
            if hasattr(args, 'tensor_parallel_size'):
                model_kwargs['tensor_parallel_size'] = args.tensor_parallel_size
            
            # Load using model loader (prefers VLLM for speed)
            model_wrapper = model_loader.load_llm(
                model_name, 
                backend="auto",  # Will choose VLLM if available, fallback to transformers
                device=args.device,
                **model_kwargs
            )
            
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            # Try without hyphen for backward compatibility
            if "Qwen-2.5" in model_name:
                model_name = model_name.replace("Qwen-2.5", "Qwen2.5")
                logger.info(f"Retrying with corrected model name: {model_name}")
                
                model_wrapper = model_loader.load_llm(
                    model_name, 
                    backend="auto",
                    device=args.device,
                    **model_kwargs
                )
            else:
                raise
        
        # For backward compatibility, still load tokenizer separately for prompt formatting
        # (VLLM handles tokenization internally, but we need it for prompt construction)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Set up device with GPU index
        if torch.cuda.is_available() and args.device != "cpu":
            device = torch.device(f"cuda:{args.gpu_index}")
        else:
            device = torch.device("cpu")
        
        if args.device == "cpu":
            model = model.to(device)
        
        # Get unique classes and create answer choices
        unique_classes = np.unique(y_train)
        
        # Create TabLLM-style template
        if template_data and 'templates' in template_data:
            template_info = next(iter(template_data['templates'].values()))
            answer_choices_str = template_info.get('answer_choices', " ||| ".join([str(cls) for cls in unique_classes]))
            answer_choices = [choice.strip() for choice in answer_choices_str.split('|||')]
            
            logger.info(f"TabLLM template found: answer_choices = {answer_choices}")
            
            # Create mapping from dataset classes to meaningful names
            class_to_name = {}
            if len(answer_choices) == len(unique_classes):
                # Map dataset classes to meaningful names in order
                # Sort unique classes to ensure consistent mapping
                sorted_classes = sorted(unique_classes)
                for i, cls in enumerate(sorted_classes):
                    if i < len(answer_choices):
                        class_to_name[cls] = answer_choices[i]
                        logger.info(f"TabLLM mapping: {cls} -> {answer_choices[i]}")
            
            # If we couldn't create a mapping, fall back to default
            if not class_to_name:
                logger.warning(f"TabLLM mapping failed: {len(answer_choices)} answer choices vs {len(unique_classes)} unique classes")
                class_to_name = {cls: str(cls) for cls in unique_classes}
                answer_choices = [str(cls) for cls in unique_classes]
            
            # Extract question from jinja template
            jinja_template = template_info.get('jinja', '')
            if 'Which of the following classes does this instance belong to' in jinja_template:
                # Extract the question part
                question_start = jinja_template.find('Which of the following')
                question_end = jinja_template.find('?') + 1
                if question_start != -1 and question_end != 0:
                    question = jinja_template[question_start:question_end]
                else:
                    question = f"Which of the following classes does this instance belong to: {', '.join(answer_choices)}?"
            else:
                question = f"Which of the following classes does this instance belong to: {', '.join(answer_choices)}?"
        else:
            # Default format following TabLLM style
            answer_choices = [str(cls) for cls in unique_classes]
            class_to_name = {cls: str(cls) for cls in unique_classes}
            question = f"Which of the following classes does this instance belong to: {', '.join(answer_choices)}?"
        
        logger.info(f"Using question: {question}")
        logger.info(f"Answer choices: {answer_choices}")
        
        # Create few-shot examples using TabLLM note format
        max_examples = args.num_few_shot_examples
        n_examples = min(max_examples, len(X_train))
        
        # Use balanced or random selection based on args
        if getattr(args, 'balanced_few_shot', False):
            # Apply balanced few-shot selection
            unique_classes = np.unique(y_train)
            n_classes = len(unique_classes)
            
            # Calculate examples per class (as evenly as possible)
            examples_per_class = n_examples // n_classes
            remainder = n_examples % n_classes
            
            example_indices = []
            for i, class_label in enumerate(unique_classes):
                class_mask = y_train == class_label
                class_indices = np.where(class_mask)[0]
                
                # Add one extra example to first 'remainder' classes
                n_select = examples_per_class + (1 if i < remainder else 0)
                n_select = min(n_select, len(class_indices))
                
                if n_select > 0:
                    selected_class_indices = np.random.RandomState(args.seed).choice(
                        class_indices, n_select, replace=False
                    )
                    example_indices.extend(selected_class_indices)
            
            example_indices = np.array(example_indices)
            logger.info(f"Using balanced few-shot selection: {len(example_indices)} examples ({examples_per_class}+ per class)")
        else:
            example_indices = np.random.choice(len(X_train), n_examples, replace=False)
            logger.info(f"Using random few-shot selection: {n_examples} examples")
        
        few_shot_examples = []
        for idx in example_indices:
            x_example = X_train.iloc[idx] if hasattr(X_train, 'iloc') else X_train[idx]
            y_example = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
            
            # Create TabLLM-style note with semantic information (use processed mapping if available)
            semantic_for_note = processed_feature_mapping.get('semantic_info') if processed_feature_mapping else None
            if semantic_for_note is None and feature_mapping:
                semantic_for_note = feature_mapping.get('semantic_info')
            note = create_tabllm_note(x_example, dataset["attribute_names"], dataset['name'], semantic_for_note)
            # Use meaningful class name if available, otherwise use the original label
            class_label = class_to_name.get(y_example, str(y_example))
            few_shot_examples.append((note, class_label))
        
        logger.info(f"Created {len(few_shot_examples)} few-shot examples (requested: {args.num_few_shot_examples})")
        
        # Make predictions using unified LLM prediction function with memory optimization
        predictions = []
        all_class_log_probs = []  # Store log probabilities for ROC AUC calculation
        completed_samples = 0
        example_inputs_outputs = []  # Store example inputs and outputs for debugging
        
        # Feature dropping mechanism for OOM handling
        dropped_features = set()  # Track which features have been dropped
        
        # Set model to eval mode for inference
        if hasattr(model, 'eval'):
            model.eval()
        
        for i in range(len(X_test)):
            # Clear GPU cache periodically to prevent memory buildup, with error handling
            if i % 10 == 0 and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError as cache_error:
                    if "assert" in str(cache_error).lower():
                        logger.warning(f"CUDA cache clear failed due to assertion error: {cache_error}. Continuing without cache clearing.")
                    else:
                        logger.warning(f"CUDA cache clear failed: {cache_error}")
                        # Continue execution anyway
            test_sample = X_test.iloc[i] if hasattr(X_test, 'iloc') else X_test[i]
            
            # Create TabLLM-style note for test sample with semantic information (use processed mapping)
            semantic_for_note = processed_feature_mapping.get('semantic_info') if processed_feature_mapping else None
            if semantic_for_note is None and feature_mapping:
                semantic_for_note = feature_mapping.get('semantic_info')
            test_note = create_tabllm_note(test_sample, dataset["attribute_names"], dataset['name'], semantic_for_note, dropped_features)
            
            # Create TabLLM-style prompt following the template format with task description
            # Add task description for better ICL
            task_description = f"Task: Given tabular data examples, classify each instance into one of the following categories: {', '.join(answer_choices)}.\n\nExamples:\n\n"
            
            # Prepare the test instance (must always be included)
            test_prompt = f"{test_note}\n\n{question}\nAnswer:"
            
            # Tokenize test prompt to ensure we have room for it
            test_tokens = tokenizer.encode(test_prompt)
            answer_tokens_estimate = 10  # Reserve tokens for the answer
            reserved_tokens = len(test_tokens) + answer_tokens_estimate + 50  # Buffer
            
            # Calculate available tokens for few-shot examples including task description
            task_description_tokens = len(tokenizer.encode(task_description))
            available_tokens = args.max_context_length - reserved_tokens - task_description_tokens
            
            # Add few-shot examples that fit within the limit
            example_parts = []
            total_example_tokens = 0
            
            num_shots = min(args.num_few_shot_examples, len(few_shot_examples))
            selected_examples = few_shot_examples[:num_shots]
            
            for note, label in selected_examples:
                example_prompt = f"{note}\n\n{question}\nAnswer: {label}"
                example_tokens = len(tokenizer.encode(example_prompt))
                
                # Check if adding this example would exceed our limit
                if total_example_tokens + example_tokens <= available_tokens:
                    example_parts.append(example_prompt)
                    total_example_tokens += example_tokens
                else:
                    # Stop adding examples if we're out of space
                    break
            
            # Construct the full prompt with task description and clear sections
            if example_parts:
                examples_section = "\n\n".join(example_parts)
                full_prompt = f"{task_description}{examples_section}\n\nNew Instance:\n{test_prompt}"
            else:
                # Fallback with no examples but still include task description
                full_prompt = f"{task_description}New Instance:\n{test_prompt}"
            
            # Log if we had to reduce examples
            if i == 0 and len(example_parts) < num_shots:
                print(f"Sample {i}: Reduced few-shot examples from {num_shots} to {len(example_parts)} due to context length limit")
            
            # Use unified prediction function with automatic fallback chain
            try:
                # Get few-shot examples in the right format for the unified function
                selected_examples = few_shot_examples[:num_shots]
                
                # Use unified prediction function
                # For now, pass the underlying model for compatibility with existing prediction functions
                # TODO: Update unified_llm_predict to work with model wrappers directly
                underlying_model = model_wrapper.get_model()
                
                prediction_result = unified_llm_predict(
                    full_prompt=full_prompt.replace(" Answer:", ""),  # Remove " Answer:" as unified function adds it
                    answer_choices=answer_choices,
                    tokenizer=tokenizer,
                    model=underlying_model,  # Pass the underlying model for compatibility
                    args=args,
                    logger=logger,
                    selected_examples=selected_examples,
                    question=question,
                    test_first_sample=(i == 0)  # Only test methods on the first sample
                )
                
                predicted_class_name = prediction_result['predicted_class']
                prediction_method = prediction_result['method']
                class_log_probs = prediction_result.get('class_log_probs', {})
                generated_text = prediction_result.get('generated_text', None)
                
                # Map back to original class value
                predicted_class = predicted_class_name  # Default fallback
                
                # Find the original class that maps to this class name
                name_to_class = {name: cls for cls, name in class_to_name.items()}
                if predicted_class_name in name_to_class:
                    predicted_class = name_to_class[predicted_class_name]
                else:
                    # Fallback: try direct string match
                    for cls in unique_classes:
                        if str(cls) == predicted_class_name:
                            predicted_class = cls
                            break
                
                predictions.append(predicted_class)
                all_class_log_probs.append(class_log_probs)
                completed_samples = i + 1
                
                # Store example inputs and outputs for first 20 samples  
                if i < 20:
                    true_label = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
                    example_inputs_outputs.append({
                        'sample_index': i,
                        'test_note': test_note,
                        'few_shot_examples': [(note, label) for note, label in selected_examples],
                        'num_shots_used': len(selected_examples),
                        'full_prompt': full_prompt if len(full_prompt) < 5000 else full_prompt[:5000] + "... [truncated]",
                        'question': question,
                        'method': prediction_method,
                        'class_log_probs': class_log_probs,
                        'generated_text': generated_text,
                        'predicted_class_name': predicted_class_name,
                        'predicted_class': predicted_class,
                        'true_class': true_label,
                        'true_class_name': class_to_name.get(true_label, str(true_label)),
                        'class_mapping': class_to_name,
                        'correct': predicted_class == true_label
                    })
                    
            except Exception as e:
                # Check if this is an OOM error
                if is_oom_error(e):
                    logger.warning(f"Unified prediction failed for sample {i} due to OOM: {e}")
                    
                    # Try dropping a feature and retry this sample
                    if drop_feature_for_oom(dropped_features, len(dataset["attribute_names"]), logger):
                        logger.info(f"Retrying sample {i} with {len(dropped_features)} dropped features")
                        # Need to regenerate few-shot examples with dropped features
                        semantic_for_regenerate = processed_feature_mapping.get('semantic_info') if processed_feature_mapping else None
                        if semantic_for_regenerate is None and feature_mapping:
                            semantic_for_regenerate = feature_mapping.get('semantic_info')
                        few_shot_examples = regenerate_few_shot_examples(
                            X_train, y_train, example_indices, 
                            dataset["attribute_names"], dataset['name'],
                            semantic_for_regenerate, dropped_features, class_to_name
                        )
                        # Continue to next iteration which will retry this sample
                        continue
                    else:
                        logger.error("No more features to drop, cannot continue")
                        # Return partial results
                        break
                else:
                    logger.warning(f"Unified prediction failed for sample {i}: {e}")
                    # Use default prediction
                    predicted_class = unique_classes[0]
                    predictions.append(predicted_class)
                    completed_samples = i + 1
            
            # Log first few predictions for debugging
            if i < 10:
                if class_log_probs:
                    print(f"Sample {i}: {dict(sorted(class_log_probs.items(), key=lambda x: x[1], reverse=True))}")
                true_label = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
                print(f"Sample {i} ({prediction_method}): Predicted: {predicted_class_name} -> {predicted_class}, True: {class_to_name.get(true_label, str(true_label))} -> {true_label}")
            elif i == 10:
                print("Done with few shot logging. Console output will be quiet.")
        
        # Calculate timing - LLMs don't have separate training, so only prediction_time and total_time
        total_time = time.time() - start_time
        prediction_time = total_time  # For LLMs, prediction time includes model loading and inference
        
        # Calculate metrics on completed samples using shared utility
        if completed_samples > 0:
            y_test_partial = y_test[:completed_samples] if hasattr(y_test, '__getitem__') else list(y_test)[:completed_samples]
            
            # Import shared metric calculation function
            from clam.utils.llm_evaluation_utils import calculate_llm_metrics
            
            # Calculate all metrics using shared function
            calculated_metrics = calculate_llm_metrics(
                y_test_partial, predictions, unique_classes, 
                all_class_log_probs, logger
            )
            
            # Extract individual metrics for backward compatibility
            accuracy = calculated_metrics['accuracy']
            balanced_acc = calculated_metrics['balanced_accuracy']
            roc_auc = calculated_metrics['roc_auc']
            f1_macro = calculated_metrics['f1_macro']
            f1_micro = calculated_metrics['f1_micro']
            f1_weighted = calculated_metrics['f1_weighted']
            precision_macro = calculated_metrics['precision_macro']
            recall_macro = calculated_metrics['recall_macro']
        else:
            accuracy = 0.0
            balanced_acc = 0.0
            roc_auc = None
            f1_macro = f1_micro = f1_weighted = None
            precision_macro = recall_macro = None
        
        results = {
            'model_name': 'TabLLM',
            'dataset_name': dataset['name'],
            'dataset_id': dataset['id'],
            'task_id': dataset['id'],  # For consistency with CLAM extraction logic
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'prediction_time': float(prediction_time),  # Time for inference (includes model loading for LLMs)
            'total_time': float(total_time),  # Same as prediction_time for LLMs (no separate training phase)
            'num_test_samples': len(X_test),
            'num_samples': len(X_train) + len(X_test),  # Total dataset size
            'completed_samples': completed_samples,
            'completion_rate': completed_samples / len(X_test) if len(X_test) > 0 else 0.0,
            'num_features': X_train.shape[1],  # Use X_train to get actual feature count after reduction
            'num_classes': len(unique_classes),
            'predictions': predictions,
            'ground_truth': (y_test[:completed_samples].tolist() if hasattr(y_test[:completed_samples], 'tolist') 
                           else list(y_test)[:completed_samples]) if completed_samples > 0 else [],
            # Additional metrics to match evaluate_on_dataset
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'f1_macro': float(f1_macro) if f1_macro is not None else None,
            'f1_micro': float(f1_micro) if f1_micro is not None else None,
            'f1_weighted': float(f1_weighted) if f1_weighted is not None else None,
            'precision_macro': float(precision_macro) if precision_macro is not None else None,
            'recall_macro': float(recall_macro) if recall_macro is not None else None,
            # TabLLM-specific metadata
            'used_template': template_data is not None,
            'used_tabllm_config': feature_mapping is not None,
            'openml_task_id': openml_task_id,
            'model_used': model_name,
            'class_mapping': class_to_name if template_data else None,
            'example_inputs_outputs': example_inputs_outputs,
            'prediction_method': 'unified',
            'feature_mapping_preserved': processed_feature_mapping is not None
        }
        
        logger.info(f"TabLLM accuracy on {dataset['name']}: {accuracy:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating TabLLM: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'model_name': 'TabLLM',
            'dataset_name': dataset['name'],
            'error': str(e)
        }