#!/usr/bin/env python
"""
CLAM-T-SNe baseline evaluation module.

This baseline uses TabPFNv2 to generate embeddings, creates t-SNE visualizations
with colored training points and grayed-out test points, then uses a Vision Language Model
to classify individual test points based on their position in the visualization.
"""

import os
import numpy as np
import torch
import time
import logging
import tempfile
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
from .model_loader import model_loader, GenerationConfig
from clam.utils.vlm_prompting import create_classification_prompt, parse_vlm_response, create_vlm_conversation
from clam.utils.class_name_utils import get_semantic_class_names_or_fallback


def convert_numpy_types(obj):
    """Convert NumPy data types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def evaluate_clam_tsne(dataset, args):
    """Evaluate CLAM-T-SNe baseline on a dataset."""
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating CLAM-T-SNe on dataset {dataset['name']}")
    
    # Import required utilities
    from clam.utils import (
        drop_feature_for_oom,
        is_oom_error,
        apply_feature_reduction
    )
    from clam.data.embeddings import get_tabpfn_embeddings
    from clam.data.tsne_visualization import (
        create_tsne_visualization,
        create_tsne_3d_visualization,
        create_combined_tsne_plot,
        create_combined_tsne_3d_plot,
        create_tsne_plot_with_knn,
        create_tsne_3d_plot_with_knn,
        encode_plot_as_base64
    )
    
    try:
        # Import VLM dependencies
        try:
            from PIL import Image
            import io
        except ImportError as e:
            logger.error(f"VLM dependencies not found: {e}")
            logger.error("Please install required packages: pip install pillow")
            return {
                'model_name': 'CLAM-T-SNe',
                'dataset_name': dataset['name'],
                'error': f"Missing dependencies: {e}"
            }
        
        # Split data
        X, y = dataset["X"], dataset["y"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed
        )
        
        # Create validation split from training data for TabPFN
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=args.seed
        )
        
        # Limit test samples if specified
        if args.max_test_samples and args.max_test_samples < len(X_test):
            X_test = X_test[:args.max_test_samples]
            y_test = y_test[:args.max_test_samples]
        
        # Apply feature selection for large feature spaces
        X_train_fit, X_test, dataset, selected_feature_indices = apply_feature_reduction(
            X_train_fit, y_train_fit, X_test, dataset, args, logger
        )
        
        # Apply same feature indices to validation set
        if selected_feature_indices is not None:
            if hasattr(X_val, 'iloc'):
                X_val = X_val.iloc[:, selected_feature_indices]
            else:
                X_val = X_val[:, selected_feature_indices]
        
        start_time = time.time()
        
        # Generate TabPFN embeddings
        logger.info("Generating TabPFN embeddings...")
        train_embeddings, val_embeddings, test_embeddings, tabpfn, y_train_sample = get_tabpfn_embeddings(
            X_train_fit, y_train_fit, X_val, X_test,
            max_samples=getattr(args, 'max_tabpfn_samples', 3000),
            embedding_size=getattr(args, 'embedding_size', 1000),
            cache_dir=getattr(args, 'cache_dir', None),
            dataset_name=dataset['name'],
            force_recompute=getattr(args, 'force_recompute_embeddings', False)
        )
        
        # Create t-SNE visualization (2D or 3D based on args)
        use_3d_tsne = getattr(args, 'use_3d_tsne', False)
        use_knn_connections = getattr(args, 'use_knn_connections', False)
        knn_k = getattr(args, 'knn_k', 5)
        
        # VLM image size configuration
        max_vlm_image_size = getattr(args, 'max_vlm_image_size', 2048)
        image_dpi = getattr(args, 'image_dpi', 100)
        force_rgb_mode = getattr(args, 'force_rgb_mode', True)
        zoom_factor = getattr(args, 'tsne_zoom_factor', 2.0)
        
        # Parse custom viewing angles if provided
        viewing_angles = None
        if use_3d_tsne and hasattr(args, 'viewing_angles') and args.viewing_angles:
            try:
                # Parse format: "elev1,azim1;elev2,azim2;..."
                angle_pairs = args.viewing_angles.split(';')
                viewing_angles = []
                for pair in angle_pairs:
                    elev, azim = map(int, pair.split(','))
                    viewing_angles.append((elev, azim))
                logger.info(f"Using custom viewing angles: {viewing_angles}")
            except Exception as e:
                logger.warning(f"Error parsing viewing angles '{args.viewing_angles}': {e}. Using defaults.")
                viewing_angles = None
        
        if use_3d_tsne:
            logger.info("Creating 3D t-SNE visualization...")
            train_tsne, test_tsne, base_fig = create_tsne_3d_visualization(
                train_embeddings, y_train_sample, test_embeddings,
                perplexity=getattr(args, 'tsne_perplexity', 30),
                n_iter=getattr(args, 'tsne_n_iter', 1000),
                random_state=args.seed
            )
        else:
            logger.info("Creating 2D t-SNE visualization...")
            train_tsne, test_tsne, base_fig = create_tsne_visualization(
                train_embeddings, y_train_sample, test_embeddings,
                perplexity=getattr(args, 'tsne_perplexity', 30),
                n_iter=getattr(args, 'tsne_n_iter', 1000),
                random_state=args.seed
            )
        
        # Close the base figure to save memory
        plt.close(base_fig)
        
        # Load VLM model using new model loader
        logger.info("Loading Vision Language Model...")
        vlm_model_id = getattr(args, 'vlm_model_id', "Qwen/Qwen2.5-VL-32B-Instruct")
        
        # Configure VLM loading parameters
        vlm_kwargs = {}
        if torch.cuda.is_available() and args.device != "cpu":
            vlm_kwargs.update({
                'torch_dtype': torch.float16,
                'device_map': "auto",
                'low_cpu_mem_usage': True
            })
        else:
            vlm_kwargs.update({
                'low_cpu_mem_usage': True
            })
        
        # Load VLM using centralized model loader with VLLM support
        backend = getattr(args, 'backend', 'auto')
        tensor_parallel_size = getattr(args, 'tensor_parallel_size', 1)
        gpu_memory_utilization = getattr(args, 'gpu_memory_utilization', 0.9)
        
        vlm_wrapper = model_loader.load_vlm(
            vlm_model_id, 
            backend=backend,
            device=args.device, 
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **vlm_kwargs
        )
        
        # Get unique classes for prompting
        unique_classes = np.unique(y_train_sample)
        
        # Extract semantic class names with fallback to Class_<NUM>
        from clam.utils.class_name_utils import extract_class_names_from_labels
        semantic_class_names, _ = extract_class_names_from_labels(
            labels=unique_classes.tolist(),
            dataset_name=dataset.get('name', None),
            semantic_data_dir=getattr(args, 'semantic_data_dir', None),
            use_semantic=getattr(args, 'use_semantic_names', False)
        )
        
        # Create mapping from numeric labels to semantic names
        class_to_semantic = {cls: name for cls, name in zip(sorted(unique_classes), semantic_class_names)}
        
        class_list_str = ", ".join([str(cls) for cls in unique_classes])
        
        # Make predictions for each test point
        predictions = []
        prediction_details = []
        completed_samples = 0
        example_inputs_outputs = []  # Store example inputs and outputs for debugging
        
        logger.info(f"Starting VLM predictions for {len(test_tsne)} test points...")
        
        for i in range(len(test_tsne)):
            try:
                # Create visualization highlighting current test point
                if use_knn_connections:
                    # Create visualization with KNN connections
                    if use_3d_tsne:
                        # Create 3D visualization with KNN connections
                        fig, legend_text, metadata = create_tsne_3d_plot_with_knn(
                            train_tsne, test_tsne, y_train_sample,
                            train_embeddings, test_embeddings,
                            highlight_test_idx=i,
                            k=knn_k,
                            figsize=(12, 9),  # Reduced from (20, 15) for VLM compatibility
                            viewing_angles=viewing_angles,
                            zoom_factor=zoom_factor
                        )
                    else:
                        # Create 2D visualization with KNN connections
                        fig, legend_text, metadata = create_tsne_plot_with_knn(
                            train_tsne, test_tsne, y_train_sample,
                            train_embeddings, test_embeddings,
                            highlight_test_idx=i,
                            k=knn_k,
                            figsize=(10, 8),  # Reduced from (12, 10) for VLM compatibility
                            zoom_factor=zoom_factor
                        )
                else:
                    # Create standard visualization without KNN connections
                    if use_3d_tsne:
                        # Create 3D visualization with multiple viewing angles
                        fig, legend_text, metadata = create_combined_tsne_3d_plot(
                            train_tsne, test_tsne, y_train_sample, 
                            highlight_test_idx=i,
                            figsize=(12, 9),  # Reduced from (20, 15) for VLM compatibility
                            viewing_angles=viewing_angles,
                            zoom_factor=zoom_factor
                        )
                    else:
                        # Create 2D visualization
                        fig, legend_text, metadata = create_combined_tsne_plot(
                            train_tsne, test_tsne, y_train_sample, 
                            highlight_test_idx=i,
                            figsize=(8, 6),  # Reduced from (10, 8) for VLM compatibility
                            zoom_factor=zoom_factor
                        )
                
                # Convert plot to image
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=image_dpi, bbox_inches='tight', facecolor='white')
                img_buffer.seek(0)
                image = Image.open(img_buffer)
                plt.close(fig)  # Clean up figure to save memory
                
                # Convert RGBA to RGB to reduce processing time
                if force_rgb_mode:
                    if image.mode == 'RGBA':
                        # Create a white background
                        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                        # Paste the RGBA image on the white background
                        rgb_image.paste(image, mask=image.split()[3] if len(image.split()) == 4 else None)
                        image = rgb_image
                        logger.debug(f"Converted image from RGBA to RGB mode")
                    elif image.mode != 'RGB':
                        # Convert any other mode to RGB
                        image = image.convert('RGB')
                        logger.debug(f"Converted image from {image.mode} to RGB mode")
                
                # Resize image if it exceeds VLM limits
                if image.width > max_vlm_image_size or image.height > max_vlm_image_size:
                    # Calculate new size maintaining aspect ratio
                    ratio = min(max_vlm_image_size / image.width, max_vlm_image_size / image.height)
                    new_width = int(image.width * ratio)
                    new_height = int(image.height * ratio)
                    logger.info(f"Resizing image from {image.width}x{image.height} to {new_width}x{new_height} for VLM compatibility")
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Determine visible classes from plot and KNN pie chart
                visible_classes = set(metadata.get('classes', []))
                
                # Add classes from KNN pie chart if available
                if metadata.get('knn_info') and 'neighbor_classes' in metadata['knn_info']:
                    knn_classes = set(metadata['knn_info']['neighbor_classes'])
                    visible_classes.update(knn_classes)
                
                # Convert to sorted list for consistent prompting
                visible_classes_list = sorted(list(visible_classes))
                
                # Map visible classes to semantic names
                visible_semantic_names = [class_to_semantic[cls] for cls in visible_classes_list]
                
                # Log the visible classes for debugging
                if i == 0:  # Only log for first sample to avoid spam
                    logger.info(f"All classes in dataset: {sorted(unique_classes.tolist())}")
                    logger.info(f"Visible classes in plot/KNN: {visible_classes_list}")
                    logger.info(f"Semantic class names: {semantic_class_names}")
                
                # Create prompt for VLM using semantic class names
                prompt = create_classification_prompt(
                    class_names=visible_semantic_names,
                    modality="tabular",
                    use_knn=use_knn_connections,
                    use_3d=use_3d_tsne,
                    knn_k=knn_k if use_knn_connections else None,
                    legend_text=legend_text,
                    dataset_description="Tabular data embedded using TabPFN features",
                    use_semantic_names=getattr(args, 'use_semantic_names', False)
                )

                # Create conversation using utility
                conversation = create_vlm_conversation(image, prompt)
                
                # Configure generation parameters
                gen_config = GenerationConfig(
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=True
                )
                
                # Generate response using the wrapper
                response = vlm_wrapper.generate_from_conversation(conversation, gen_config)
                
                # Parse prediction from response using semantic names
                # Note: parse_vlm_response expects the same format used in the prompt
                prediction = parse_vlm_response(response, visible_semantic_names, logger, use_semantic_names=True)
                
                # Map back to numeric label if needed
                if prediction in visible_semantic_names:
                    semantic_to_numeric = {name: cls for cls, name in class_to_semantic.items() if cls in visible_classes_list}
                    prediction = semantic_to_numeric.get(prediction, prediction)
                predictions.append(prediction)
                
                # Store details for debugging
                true_label = y_test[i] if hasattr(y_test, '__getitem__') else y_test.iloc[i]
                prediction_details.append({
                    'test_point_idx': i,
                    'vlm_response': response,
                    'parsed_prediction': prediction,
                    'true_label': true_label,
                    'tsne_coords': test_tsne[i].tolist(),
                    'image_size': f"{image.width}x{image.height}",
                    'image_mode': image.mode,
                    'all_classes': unique_classes.tolist(),
                    'visible_classes': visible_classes_list,
                    'n_visible_classes': len(visible_classes_list)
                })
                
                # Store example inputs and outputs for first 20 samples
                if i < 20:
                    example_inputs_outputs.append({
                        'sample_index': i,
                        'test_point_coords': test_tsne[i].tolist(),
                        'image_size': f"{image.width}x{image.height}",
                        'image_mode': image.mode,
                        'prompt': prompt if len(prompt) < 2000 else prompt[:2000] + "... [truncated]",
                        'vlm_model': vlm_model_id,
                        'vlm_response': response,
                        'parsed_prediction': prediction,
                        'true_class': true_label,
                        'correct': prediction == true_label,
                        'tsne_params': {
                            'use_3d': use_3d_tsne,
                            'use_knn_connections': use_knn_connections,
                            'knn_k': knn_k if use_knn_connections else None,
                            'viewing_angles': viewing_angles if use_3d_tsne else None
                        }
                    })
                
                completed_samples = i + 1
                
                # Log first few predictions for debugging
                if i < 10:
                    print(f"Sample {i}: Predicted: {prediction}, True: {true_label}, Correct: {prediction == true_label}")
                    logger.info(f"Sample {i}: Predicted: {prediction}, True: {true_label}, Correct: {prediction == true_label}")
                elif i == 10:
                    print("Done with detailed prediction logging. Console output will be quieter.")
                    logger.info("Done with detailed prediction logging. Console output will be quieter.")
                
                # Log progress periodically
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(test_tsne)} predictions")
                
                # Clear memory periodically
                if (i + 1) % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as pred_error:
                logger.error(f"VLM prediction failed for test point {i}: {pred_error}")
                logger.error(f"Error type: {type(pred_error).__name__}")
                logger.error(f"Image type: {type(image)}, Image mode: {getattr(image, 'mode', 'N/A')}")
                logger.error(f"Prompt length: {len(prompt)} chars")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Use random prediction as fallback
                prediction = np.random.choice(unique_classes)
                predictions.append(prediction)
                true_label = y_test[i] if hasattr(y_test, '__getitem__') else y_test.iloc[i]
                prediction_details.append({
                    'test_point_idx': i,
                    'vlm_response': f"ERROR: {pred_error}",
                    'parsed_prediction': prediction,
                    'true_label': true_label,
                    'tsne_coords': test_tsne[i].tolist(),
                    'image_size': f"{getattr(image, 'width', 'N/A')}x{getattr(image, 'height', 'N/A')}",
                    'image_mode': getattr(image, 'mode', 'N/A')
                })
                
                # Store example inputs and outputs for first 20 samples (even on error)
                if i < 20:
                    example_inputs_outputs.append({
                        'sample_index': i,
                        'test_point_coords': test_tsne[i].tolist() if 'test_tsne' in locals() else None,
                        'image_size': f"{getattr(image, 'width', 'N/A')}x{getattr(image, 'height', 'N/A')}",
                        'image_mode': getattr(image, 'mode', 'N/A'),
                        'prompt': prompt if 'prompt' in locals() else 'N/A',
                        'vlm_model': vlm_model_id,
                        'vlm_response': f"ERROR: {pred_error}",
                        'parsed_prediction': prediction,
                        'true_class': true_label,
                        'correct': prediction == true_label,
                        'error': str(pred_error),
                        'tsne_params': {
                            'use_3d': use_3d_tsne,
                            'use_knn_connections': use_knn_connections,
                            'knn_k': knn_k if use_knn_connections else None,
                            'viewing_angles': viewing_angles if use_3d_tsne else None
                        }
                    })
                
                completed_samples = i + 1
        
        # Calculate metrics on completed samples
        if completed_samples > 0:
            # Convert predictions to same type as ground truth
            predictions_converted = []
            y_test_partial = y_test[:completed_samples] if hasattr(y_test, '__getitem__') else list(y_test)[:completed_samples]
            target_type = type(y_test_partial[0])
            
            for pred in predictions:
                try:
                    if target_type == int:
                        converted_pred = int(pred)
                    elif target_type == float:
                        converted_pred = float(pred)
                    elif target_type == str:
                        converted_pred = str(pred)
                    else:
                        converted_pred = target_type(pred)
                    predictions_converted.append(converted_pred)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert prediction '{pred}' to type {target_type}: {e}")
                    predictions_converted.append(pred)
            
            # Import shared metric calculation function
            from clam.utils.llm_evaluation_utils import calculate_llm_metrics
            
            # Calculate all metrics using shared function
            calculated_metrics = calculate_llm_metrics(
                y_test_partial, predictions_converted, unique_classes, 
                all_class_log_probs=None, logger=logger
            )
            
            # Extract individual metrics
            accuracy = calculated_metrics['accuracy']
            balanced_acc = calculated_metrics['balanced_accuracy']
            roc_auc = calculated_metrics['roc_auc']
            f1_macro = calculated_metrics['f1_macro']
            f1_micro = calculated_metrics['f1_micro']
            f1_weighted = calculated_metrics['f1_weighted']
            precision_macro = calculated_metrics['precision_macro']
            recall_macro = calculated_metrics['recall_macro']
        else:
            predictions_converted = predictions
            accuracy = 0.0
            balanced_acc = 0.0
            roc_auc = None
            f1_macro = f1_micro = f1_weighted = None
            precision_macro = recall_macro = None
        
        # Calculate timing - LLMs don't have separate training, so only prediction_time and total_time
        total_time = time.time() - start_time
        prediction_time = total_time  # For LLMs, prediction time includes model loading and inference
        
        results = {
            'model_name': 'CLAM-T-SNe',
            'dataset_name': dataset['name'],
            'dataset_id': dataset['id'],
            'task_id': dataset['id'],  # For consistency with CLAM extraction logic
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'prediction_time': float(prediction_time),  # Time for inference (includes model loading for LLMs)
            'total_time': float(total_time),  # Same as prediction_time for LLMs (no separate training phase)
            'num_test_samples': len(X_test),
            'num_samples': len(X_train) + len(X_test),
            'completed_samples': completed_samples,
            'completion_rate': completed_samples / len(X_test) if len(X_test) > 0 else 0.0,
            'num_features': X_train_fit.shape[1],
            'num_classes': len(unique_classes),
            'predictions': predictions_converted,
            'ground_truth': (y_test[:completed_samples].tolist() if hasattr(y_test[:completed_samples], 'tolist') 
                           else list(y_test)[:completed_samples]) if completed_samples > 0 else [],
            # Additional metrics
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'f1_macro': float(f1_macro) if f1_macro is not None else None,
            'f1_micro': float(f1_micro) if f1_micro is not None else None,
            'f1_weighted': float(f1_weighted) if f1_weighted is not None else None,
            'precision_macro': float(precision_macro) if precision_macro is not None else None,
            'recall_macro': float(recall_macro) if recall_macro is not None else None,
            # CLAM-T-SNe specific metadata
            'vlm_model_id': vlm_model_id,
            'tsne_params': {
                'perplexity': getattr(args, 'tsne_perplexity', 30),
                'n_iter': getattr(args, 'tsne_n_iter', 1000),
                'use_3d': use_3d_tsne,
                'viewing_angles': viewing_angles if use_3d_tsne else None,
                'use_knn_connections': use_knn_connections,
                'knn_k': knn_k if use_knn_connections else None,
                'zoom_factor': zoom_factor
            },
            'image_params': {
                'max_vlm_image_size': max_vlm_image_size,
                'image_dpi': image_dpi,
                'force_rgb_mode': force_rgb_mode
            },
            'embedding_size': getattr(args, 'embedding_size', 1000),
            'prediction_details': prediction_details[:20],  # Store first 20 for debugging
            'example_inputs_outputs': example_inputs_outputs  # Store detailed info for first 20 samples
        }
        
        # Save sample t-SNE visualizations for debugging and documentation
        save_sample_visualizations = getattr(args, 'save_sample_visualizations', True)
        if save_sample_visualizations and hasattr(args, 'output_dir') and args.output_dir:
            try:
                logger.info("Saving sample t-SNE visualizations...")
                
                # Create visualizations directory
                viz_dir = os.path.join(args.output_dir, 'tsne_visualizations', dataset['name'])
                os.makedirs(viz_dir, exist_ok=True)
                
                # Save a few sample visualizations with different test points highlighted
                num_samples_to_save = min(5, len(test_tsne), completed_samples)
                sample_indices = np.linspace(0, min(len(test_tsne)-1, completed_samples-1), num_samples_to_save, dtype=int)
                
                for i, test_idx in enumerate(sample_indices):
                    try:
                        # Create visualization for this sample
                        if use_knn_connections:
                            # Create visualization with KNN connections
                            if use_3d_tsne:
                                # Create 3D visualization with KNN connections
                                fig, legend_text, metadata = create_tsne_3d_plot_with_knn(
                                    train_tsne, test_tsne, y_train_sample,
                                    train_embeddings, test_embeddings,
                                    highlight_test_idx=test_idx,
                                    k=knn_k,
                                    figsize=(12, 9),
                                    viewing_angles=viewing_angles,
                                    zoom_factor=zoom_factor
                                )
                            else:
                                # Create 2D visualization with KNN connections
                                fig, legend_text, metadata = create_tsne_plot_with_knn(
                                    train_tsne, test_tsne, y_train_sample,
                                    train_embeddings, test_embeddings,
                                    highlight_test_idx=test_idx,
                                    k=knn_k,
                                    figsize=(10, 8),
                                    zoom_factor=zoom_factor
                                )
                        else:
                            # Create standard visualization without KNN connections
                            if use_3d_tsne:
                                # Create 3D visualization with multiple viewing angles
                                fig, legend_text, metadata = create_combined_tsne_3d_plot(
                                    train_tsne, test_tsne, y_train_sample, 
                                    highlight_test_idx=test_idx,
                                    figsize=(12, 9),
                                    viewing_angles=viewing_angles,
                                    zoom_factor=zoom_factor
                                )
                            else:
                                # Create 2D visualization
                                fig, legend_text, metadata = create_combined_tsne_plot(
                                    train_tsne, test_tsne, y_train_sample, 
                                    highlight_test_idx=test_idx,
                                    figsize=(8, 6),
                                    zoom_factor=zoom_factor
                                )
                        
                        # Save the visualization
                        sample_filename = f"sample_{test_idx:03d}_{'3d' if use_3d_tsne else '2d'}_tsne"
                        if use_knn_connections:
                            sample_filename += f"_knn{knn_k}"
                        sample_filename += ".png"
                        
                        sample_path = os.path.join(viz_dir, sample_filename)
                        fig.savefig(sample_path, dpi=image_dpi, bbox_inches='tight', facecolor='white')
                        plt.close(fig)
                        
                        # Save metadata for this visualization
                        metadata_filename = f"sample_{test_idx:03d}_metadata.json"
                        metadata_path = os.path.join(viz_dir, metadata_filename)
                        
                        true_label = y_test[test_idx] if hasattr(y_test, '__getitem__') else y_test.iloc[test_idx]
                        pred_label = predictions_converted[test_idx] if test_idx < len(predictions_converted) else None
                        
                        # Find VLM response for this test point
                        vlm_response = None
                        for detail in prediction_details:
                            if detail.get('test_point_idx') == test_idx:
                                vlm_response = detail.get('vlm_response')
                                break
                        
                        sample_metadata = {
                            'test_point_idx': int(test_idx),
                            'tsne_coords': test_tsne[test_idx].tolist(),
                            'true_label': convert_numpy_types(true_label),
                            'predicted_label': convert_numpy_types(pred_label),
                            'correct_prediction': bool(pred_label == true_label) if pred_label is not None else None,
                            'vlm_raw_response': vlm_response,
                            'legend_text': legend_text,
                            'visualization_metadata': convert_numpy_types(metadata),
                            'image_params': {
                                'dpi': image_dpi,
                                'max_size': max_vlm_image_size,
                                'force_rgb': force_rgb_mode
                            },
                            'tsne_params': {
                                'use_3d': use_3d_tsne,
                                'use_knn_connections': use_knn_connections,
                                'knn_k': knn_k if use_knn_connections else None,
                                'viewing_angles': viewing_angles if use_3d_tsne else None
                            }
                        }
                        
                        with open(metadata_path, 'w') as f:
                            json.dump(convert_numpy_types(sample_metadata), f, indent=2)
                        
                        logger.debug(f"Saved sample visualization: {sample_path}")
                        
                    except Exception as viz_error:
                        logger.warning(f"Failed to save sample visualization for test point {test_idx}: {viz_error}")
                
                # Also save the base visualization without any highlighted points
                try:
                    if use_3d_tsne:
                        base_fig, base_legend, base_metadata = create_combined_tsne_3d_plot(
                            train_tsne, test_tsne, y_train_sample,
                            figsize=(12, 9),
                            viewing_angles=viewing_angles
                        )
                    else:
                        base_fig, base_legend, base_metadata = create_combined_tsne_plot(
                            train_tsne, test_tsne, y_train_sample,
                            figsize=(8, 6)
                        )
                    
                    base_filename = f"overview_{'3d' if use_3d_tsne else '2d'}_tsne.png"
                    base_path = os.path.join(viz_dir, base_filename)
                    base_fig.savefig(base_path, dpi=image_dpi, bbox_inches='tight', facecolor='white')
                    plt.close(base_fig)
                    
                    # Save overview metadata
                    overview_metadata = {
                        'dataset_name': dataset['name'],
                        'dataset_id': dataset['id'],
                        'task_id': dataset['id'],  # For consistency with CLAM extraction logic
                        'num_train_points': len(train_tsne),
                        'num_test_points': len(test_tsne),
                        'num_classes': len(unique_classes),
                        'classes': convert_numpy_types(unique_classes.tolist()),
                        'legend_text': base_legend,
                        'visualization_metadata': convert_numpy_types(base_metadata),
                        'tsne_params': {
                            'perplexity': getattr(args, 'tsne_perplexity', 30),
                            'n_iter': getattr(args, 'tsne_n_iter', 1000),
                            'use_3d': use_3d_tsne,
                            'use_knn_connections': use_knn_connections,
                            'knn_k': knn_k if use_knn_connections else None,
                            'viewing_angles': viewing_angles if use_3d_tsne else None,
                            'zoom_factor': zoom_factor
                        }
                    }
                    
                    overview_metadata_path = os.path.join(viz_dir, "overview_metadata.json")
                    with open(overview_metadata_path, 'w') as f:
                        json.dump(convert_numpy_types(overview_metadata), f, indent=2)
                    
                    logger.info(f"Saved overview t-SNE visualization: {base_path}")
                    
                except Exception as overview_error:
                    logger.warning(f"Failed to save overview visualization: {overview_error}")
                
                # Add visualization info to results
                results['saved_visualizations'] = {
                    'visualization_dir': viz_dir,
                    'num_sample_visualizations': num_samples_to_save,
                    'sample_indices': sample_indices.tolist(),
                    'overview_saved': True
                }
                
                logger.info(f"Saved {num_samples_to_save} sample t-SNE visualizations in: {viz_dir}")
                
            except Exception as save_error:
                logger.warning(f"Failed to save t-SNE visualizations: {save_error}")
                results['visualization_save_error'] = str(save_error)
        
        # Log artifacts to wandb if available
        try:
            import wandb
            if wandb.run is not None:
                logger.info("Logging artifacts to wandb...")
                
                # Create a temporary JSON file with example inputs/outputs for wandb
                if example_inputs_outputs:
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        wandb_data = {
                            'dataset_name': dataset['name'],
                            'dataset_id': dataset['id'],
                            'task_id': dataset['id'],  # For consistency with CLAM extraction logic
                            'model_name': 'CLAM-T-SNe',
                            'accuracy': float(accuracy),
                            'num_samples': len(example_inputs_outputs),
                            'example_inputs_outputs': convert_numpy_types(example_inputs_outputs),
                            'tsne_params': {
                                'use_3d': use_3d_tsne,
                                'use_knn_connections': use_knn_connections,
                                'knn_k': knn_k if use_knn_connections else None,
                                'viewing_angles': viewing_angles if use_3d_tsne else None
                            }
                        }
                        json.dump(convert_numpy_types(wandb_data), f, indent=2)
                        json_temp_path = f.name
                    
                    # Log the JSON file as an artifact
                    wandb.log_artifact(json_temp_path, name=f"clam_tsne_examples_{dataset['name']}", type="examples")
                    
                    # Clean up temp file
                    os.unlink(json_temp_path)
                    logger.info("Logged example inputs/outputs JSON to wandb")
                
                # Log visualizations if they were saved
                if save_sample_visualizations and 'saved_visualizations' in results:
                    viz_dir = results['saved_visualizations']['visualization_dir']
                    
                    # Log sample visualizations
                    sample_images = []
                    for idx in results['saved_visualizations']['sample_indices']:
                        sample_filename = f"sample_{idx:03d}_{'3d' if use_3d_tsne else '2d'}_tsne"
                        if use_knn_connections:
                            sample_filename += f"_knn{knn_k}"
                        sample_filename += ".png"
                        
                        sample_path = os.path.join(viz_dir, sample_filename)
                        if os.path.exists(sample_path):
                            # Read the corresponding metadata for caption
                            metadata_path = os.path.join(viz_dir, f"sample_{idx:03d}_metadata.json")
                            caption = f"Test point {idx}"
                            if os.path.exists(metadata_path):
                                try:
                                    with open(metadata_path, 'r') as f:
                                        metadata = json.load(f)
                                    true_label = metadata.get('true_label', 'Unknown')
                                    pred_label = metadata.get('predicted_label', 'Unknown')
                                    correct = metadata.get('correct_prediction', False)
                                    caption = f"Test point {idx}: True={true_label}, Pred={pred_label}, Correct={correct}"
                                except Exception as e:
                                    logger.warning(f"Could not read metadata for sample {idx}: {e}")
                            
                            sample_images.append(wandb.Image(sample_path, caption=caption))
                    
                    # Log overview visualization
                    overview_filename = f"overview_{'3d' if use_3d_tsne else '2d'}_tsne.png"
                    overview_path = os.path.join(viz_dir, overview_filename)
                    if os.path.exists(overview_path):
                        overview_caption = f"t-SNE Overview ({dataset['name']}) - {'3D' if use_3d_tsne else '2D'}"
                        if use_knn_connections:
                            overview_caption += f" with KNN (k={knn_k})"
                        sample_images.append(wandb.Image(overview_path, caption=overview_caption))
                    
                    # Log all images
                    if sample_images:
                        wandb.log({
                            f"tsne_visualizations/{dataset['name']}": sample_images,
                            f"num_visualizations/{dataset['name']}": len(sample_images)
                        })
                        logger.info(f"Logged {len(sample_images)} t-SNE visualizations to wandb")
                
                # Log key metrics for this dataset
                wandb.log({
                    f"dataset_metrics/{dataset['name']}/accuracy": float(accuracy),
                    f"dataset_metrics/{dataset['name']}/balanced_accuracy": float(balanced_acc),
                    f"dataset_metrics/{dataset['name']}/completed_samples": completed_samples,
                    f"dataset_metrics/{dataset['name']}/completion_rate": completed_samples / len(X_test) if len(X_test) > 0 else 0.0,
                    f"dataset_metrics/{dataset['name']}/num_test_samples": len(X_test),
                    f"dataset_metrics/{dataset['name']}/num_classes": len(unique_classes),
                    f"dataset_metrics/{dataset['name']}/roc_auc": float(roc_auc) if roc_auc is not None else None,
                })
                
        except ImportError:
            logger.debug("wandb not available, skipping artifact logging")
        except Exception as wandb_error:
            logger.warning(f"Failed to log artifacts to wandb: {wandb_error}")
        
        logger.info(f"CLAM-T-SNe accuracy on {dataset['name']}: {accuracy:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating CLAM-T-SNe: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'model_name': 'CLAM-T-SNe',
            'dataset_name': dataset['name'],
            'error': str(e)
        }


