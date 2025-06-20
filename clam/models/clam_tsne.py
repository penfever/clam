#!/usr/bin/env python
"""
CLAM t-SNE classifier - A unified implementation for tabular, audio, and vision modalities.

This module provides a centralized CLAM t-SNE classifier that can work across different
data modalities by using appropriate embedding methods and t-SNE visualizations with
Vision Language Model classification.
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
from typing import Dict, Any, Optional, List, Tuple, Union
import matplotlib.pyplot as plt

# Import unified model loader for VLM
from clam.utils.model_loader import model_loader, GenerationConfig

# Import VLM prompting utilities
from clam.utils.vlm_prompting import create_classification_prompt, parse_vlm_response, create_vlm_conversation
from clam.utils.class_name_utils import extract_class_names_from_labels

# Import new multi-visualization framework
from clam.viz import ContextComposer, VisualizationConfig
from clam.viz.context.layouts import LayoutStrategy
from clam.viz.context.composer import CompositionConfig


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


class ClamTsneClassifier:
    """
    Unified CLAM t-SNE classifier for tabular, audio, and vision data.
    
    This classifier:
    1. Generates embeddings using modality-specific methods (TabPFN, Whisper, CLAP, etc.)
    2. Creates t-SNE visualizations with training and test points
    3. Uses a Vision Language Model to classify test points based on their position
    """
    
    def __init__(
        self,
        modality: str = "tabular",
        vlm_model_id: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        embedding_size: int = 1000,
        tsne_perplexity: int = 30,
        tsne_n_iter: int = 1000,
        use_3d_tsne: bool = False,
        use_knn_connections: bool = False,
        knn_k: int = 5,
        max_vlm_image_size: int = 2048,
        image_dpi: int = 100,
        force_rgb_mode: bool = True,
        tsne_zoom_factor: float = 2.0,
        max_tabpfn_samples: int = 3000,
        cache_dir: Optional[str] = None,
        use_semantic_names: bool = False,
        device: str = "auto",
        backend: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        enable_thinking: bool = True,
        openai_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
        api_model: Optional[str] = None,
        seed: int = 42,
        # New multi-visualization parameters
        enable_multi_viz: bool = False,
        visualization_methods: Optional[List[str]] = None,
        layout_strategy: str = "adaptive_grid",
        reasoning_focus: str = "classification",
        multi_viz_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize CLAM t-SNE classifier.
        
        Args:
            modality: Data modality ("tabular", "audio", "vision")
            vlm_model_id: Vision Language Model identifier (for local models)
            embedding_size: Size of embeddings for TabPFN
            tsne_perplexity: t-SNE perplexity parameter
            tsne_n_iter: Number of t-SNE iterations
            use_3d_tsne: Whether to use 3D t-SNE visualization
            use_knn_connections: Whether to show KNN connections
            knn_k: Number of nearest neighbors for KNN
            max_vlm_image_size: Maximum image size for VLM
            image_dpi: DPI for generated images
            force_rgb_mode: Whether to force RGB mode for images
            tsne_zoom_factor: Zoom factor for t-SNE plots
            max_tabpfn_samples: Maximum samples for TabPFN (tabular only)
            cache_dir: Directory for caching embeddings
            use_semantic_names: Whether to use semantic class names
            device: Device for computation
            backend: Backend for VLM loading
            tensor_parallel_size: Tensor parallel size for distributed inference
            gpu_memory_utilization: GPU memory utilization factor
            enable_thinking: Enable thinking mode for compatible API models
            openai_model: OpenAI model identifier (e.g., 'gpt-4o', 'gpt-4.1')
            gemini_model: Gemini model identifier (e.g., 'gemini-2.5-pro', 'gemini-2.5-flash')
            api_model: Generic API model identifier (auto-detects provider)
            seed: Random seed
            enable_multi_viz: Whether to use multi-visualization framework (default: False for backward compatibility)
            visualization_methods: List of visualization methods to use (e.g., ['tsne', 'pca', 'umap'])
            layout_strategy: Layout strategy for multi-visualization composition
            reasoning_focus: Focus for multi-visualization reasoning (classification, comparison, etc.)
            multi_viz_config: Additional configuration for multi-visualization
            **kwargs: Additional modality-specific arguments
        """
        self.modality = modality.lower()
        self.vlm_model_id = vlm_model_id
        self.embedding_size = embedding_size
        self.tsne_perplexity = tsne_perplexity
        self.tsne_n_iter = tsne_n_iter
        self.use_3d_tsne = use_3d_tsne
        self.use_knn_connections = use_knn_connections
        self.knn_k = knn_k
        self.max_vlm_image_size = max_vlm_image_size
        self.image_dpi = image_dpi
        self.force_rgb_mode = force_rgb_mode
        self.tsne_zoom_factor = tsne_zoom_factor
        self.max_tabpfn_samples = max_tabpfn_samples
        self.cache_dir = cache_dir
        self.use_semantic_names = use_semantic_names
        self.device = device
        self.backend = backend
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_thinking = enable_thinking
        self.openai_model = openai_model
        self.gemini_model = gemini_model
        self.api_model = api_model
        self.seed = seed
        
        # New multi-visualization parameters
        self.enable_multi_viz = enable_multi_viz
        self.visualization_methods = visualization_methods or ['tsne']
        self.layout_strategy = layout_strategy
        self.reasoning_focus = reasoning_focus
        self.multi_viz_config = multi_viz_config or {}
        
        # Determine the actual model to use (API models take precedence)
        self.effective_model_id = self._determine_effective_model()
        self.is_api_model = self._is_api_model(self.effective_model_id)
        
        # Store additional kwargs for modality-specific parameters
        self.modality_kwargs = kwargs
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize VLM wrapper (loaded lazily)
        self.vlm_wrapper = None
        
        # Store fitted data
        self.train_embeddings = None
        self.test_embeddings = None
        self.train_tsne = None
        self.test_tsne = None
        self.y_train_sample = None
        self.class_names = None
        self.unique_classes = None
        self.class_to_semantic = None
        
        # Multi-visualization context composer
        self.context_composer = None
    
    def _determine_effective_model(self) -> str:
        """Determine the effective model ID to use based on API model parameters."""
        # Priority order: api_model > openai_model > gemini_model > vlm_model_id
        if self.api_model:
            return self.api_model
        elif self.openai_model:
            return self.openai_model
        elif self.gemini_model:
            return self.gemini_model
        else:
            return self.vlm_model_id
    
    def _is_api_model(self, model_id: str) -> bool:
        """Check if the model ID corresponds to an API model."""
        api_model_patterns = [
            # OpenAI models
            'gpt-4', 'gpt-3.5', 'gpt-4o', 'gpt-4.1',
            # Gemini models
            'gemini-', 'gemini-2.', 'gemini-2.5', 'gemini-2.0'
        ]
        return any(pattern in model_id.lower() for pattern in api_model_patterns)
        
    def _get_embedding_method(self):
        """Get the appropriate embedding method for the modality."""
        if self.modality == "tabular":
            from clam.data.embeddings import get_tabpfn_embeddings
            return get_tabpfn_embeddings
        elif self.modality == "audio":
            # Audio embeddings will be handled by specific audio classifier implementations
            raise NotImplementedError("Use ClamAudioTsneClassifier for audio data")
        elif self.modality == "vision":
            # Vision embeddings will be handled by specific vision classifier implementations  
            raise NotImplementedError("Use ClamImageTsneClassifier for vision data")
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")
    
    def _get_tsne_visualization_methods(self):
        """Get t-SNE visualization methods."""
        from clam.data.tsne_visualization import (
            create_tsne_visualization,
            create_tsne_3d_visualization,
            create_combined_tsne_plot,
            create_combined_tsne_3d_plot,
            create_tsne_plot_with_knn,
            create_tsne_3d_plot_with_knn
        )
        return {
            'create_tsne_visualization': create_tsne_visualization,
            'create_tsne_3d_visualization': create_tsne_3d_visualization,
            'create_combined_tsne_plot': create_combined_tsne_plot,
            'create_combined_tsne_3d_plot': create_combined_tsne_3d_plot,
            'create_tsne_plot_with_knn': create_tsne_plot_with_knn,
            'create_tsne_3d_plot_with_knn': create_tsne_3d_plot_with_knn
        }
    
    def _load_vlm(self):
        """Load the Vision Language Model (local or API-based)."""
        if self.vlm_wrapper is not None:
            return self.vlm_wrapper
            
        model_to_load = self.effective_model_id
        self.logger.info(f"Loading Vision Language Model: {model_to_load}")
        
        if self.is_api_model:
            # API model - minimal configuration needed
            self.logger.info("Using API-based VLM (OpenAI/Gemini)")
            vlm_kwargs = {}
            
            # For API models, backend is auto-detected by model_loader
            backend = "auto"
        else:
            # Local model - configure hardware parameters
            self.logger.info("Using local VLM")
            vlm_kwargs = {}
            if torch.cuda.is_available() and self.device != "cpu":
                vlm_kwargs.update({
                    'torch_dtype': torch.float16,
                    'device_map': "auto",
                    'low_cpu_mem_usage': True
                })
            else:
                vlm_kwargs.update({
                    'low_cpu_mem_usage': True
                })
            
            backend = self.backend
        
        # Load VLM using centralized model loader
        self.vlm_wrapper = model_loader.load_vlm(
            model_to_load,
            backend=backend,
            device=self.device if not self.is_api_model else None,
            tensor_parallel_size=self.tensor_parallel_size if not self.is_api_model else None,
            gpu_memory_utilization=self.gpu_memory_utilization if not self.is_api_model else None,
            **vlm_kwargs
        )
        
        return self.vlm_wrapper
    
    def _initialize_multi_viz_composer(self, X_train, y_train, X_test=None):
        """Initialize the multi-visualization context composer."""
        if not self.enable_multi_viz:
            return
            
        self.logger.info("Initializing multi-visualization context composer...")
        
        # Create composition configuration
        config = CompositionConfig(
            layout_strategy=LayoutStrategy[self.layout_strategy.upper()],
            reasoning_focus=self.reasoning_focus,
            optimize_for_vlm=True
        )
        
        # Initialize context composer
        self.context_composer = ContextComposer(config)
        
        # Add visualizations based on specified methods
        for viz_method in self.visualization_methods:
            viz_config = VisualizationConfig(
                use_3d=self.use_3d_tsne,
                title=f"{viz_method.upper()} - {self.modality.title()} Data",
                random_state=self.seed,
                figsize=(8, 6),
                point_size=50
            )
            
            # Get method-specific configuration
            method_config = self.multi_viz_config.get(viz_method, {})
            
            # Add method-specific parameters based on visualization type
            if viz_method == 'tsne':
                method_config.update({
                    'perplexity': min(self.tsne_perplexity, len(X_train) // 4),
                    'max_iter': self.tsne_n_iter
                })
            elif viz_method == 'umap':
                method_config.setdefault('n_neighbors', 15)
                method_config.setdefault('min_dist', 0.1)
            elif viz_method == 'spectral':
                method_config.update({
                    'n_neighbors': max(2, len(X_train) // 20),
                    'affinity': 'nearest_neighbors'
                })
            elif viz_method == 'isomap':
                method_config.setdefault('n_neighbors', 10)
                
            self.context_composer.add_visualization(
                viz_method,
                config=method_config,
                viz_config=viz_config
            )
        
        self.logger.info(f"Added {len(self.context_composer.visualizations)} visualization methods")
        
        # Fit all visualizations
        self.context_composer.fit(X_train, y_train, X_test)
    
    def fit(self, X_train, y_train, X_test=None, class_names=None, task_type=None, **kwargs):
        """
        Fit the CLAM t-SNE model for both classification and regression.
        
        Args:
            X_train: Training features
            y_train: Training labels/targets
            X_test: Test features (optional, for creating visualizations)
            class_names: Class names (optional, for classification)
            task_type: 'classification' or 'regression' (auto-detected if None)
            **kwargs: Additional arguments
        """
        self.logger.info(f"Fitting CLAM t-SNE model for {self.modality} data...")
        
        # Handle different input formats
        if hasattr(X_train, 'values'):
            X_train_array = X_train.values
        else:
            X_train_array = np.array(X_train)
            
        if hasattr(y_train, 'values'):
            y_train_array = y_train.values
        else:
            y_train_array = np.array(y_train)
        
        # Detect task type
        try:
            from clam.utils.task_detection import detect_task_type, get_target_statistics
            self.task_type, detection_method = detect_task_type(
                y=y_train_array, 
                manual_override=task_type,
                dataset_info=kwargs.get('dataset_info')
            )
            self.logger.info(f"Task type: {self.task_type} (detected via: {detection_method})")
            
            # Get target statistics for regression or class info for classification
            if self.task_type == 'regression':
                self.target_stats = get_target_statistics(y_train_array)
                self.logger.info(f"Target statistics: {self.target_stats}")
            else:
                self.unique_classes = np.unique(y_train_array)
                self.target_stats = None
        except ImportError:
            # Fallback if task detection is not available
            self.logger.warning("Task detection not available, assuming classification")
            self.task_type = 'classification'
            self.unique_classes = np.unique(y_train_array)
            self.target_stats = None
        
        # Apply feature reduction for tabular data if needed
        if self.modality == "tabular":
            from clam.utils import apply_feature_reduction
            # Create a mock dataset dict for feature reduction
            mock_dataset = {'name': 'training_data'}
            mock_args = type('Args', (), {
                'feature_selection_threshold': getattr(self, 'feature_selection_threshold', 500),
                'seed': self.seed
            })()
            
            X_train_array, X_test_array, _, self.selected_feature_indices = apply_feature_reduction(
                X_train_array, y_train_array, 
                X_test.values if X_test is not None and hasattr(X_test, 'values') else X_test,
                mock_dataset, mock_args, self.logger
            )
            
            if X_test is not None and self.selected_feature_indices is not None:
                X_test = X_test_array
        
        # Create validation split for TabPFN
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train_array, y_train_array, test_size=0.2, random_state=self.seed
        )
        
        # Generate embeddings using modality-specific method
        if self.modality == "tabular":
            embedding_method = self._get_embedding_method()
            
            # Prepare test data for embedding
            if X_test is not None:
                X_test_for_embedding = X_test
            else:
                # Use a small subset of training data as test for visualization
                X_test_for_embedding = X_train_fit[:5]
                
            self.train_embeddings, self.val_embeddings, self.test_embeddings, self.tabpfn, self.y_train_sample = embedding_method(
                X_train_fit, y_train_fit, X_val, X_test_for_embedding,
                max_samples=self.max_tabpfn_samples,
                embedding_size=self.embedding_size,
                cache_dir=self.cache_dir,
                dataset_name='training_data',
                force_recompute=getattr(self, 'force_recompute_embeddings', False),
                task_type=self.task_type,
                seed=self.seed
            )
        else:
            raise NotImplementedError(f"Embedding generation not implemented for {self.modality}")
        
        # Create t-SNE visualization based on task type
        viz_methods = self._get_tsne_visualization_methods()
        
        if self.task_type == 'regression':
            # Use regression-specific visualization methods
            if self.use_3d_tsne:
                self.logger.info("Creating 3D regression t-SNE visualization...")
                self.train_tsne, self.test_tsne, base_fig = viz_methods['create_regression_tsne_3d_visualization'](
                    self.train_embeddings, self.y_train_sample, self.test_embeddings,
                    perplexity=self.tsne_perplexity,
                    n_iter=self.tsne_n_iter,
                    random_state=self.seed
                )
            else:
                self.logger.info("Creating 2D regression t-SNE visualization...")
                self.train_tsne, self.test_tsne, base_fig = viz_methods['create_regression_tsne_visualization'](
                    self.train_embeddings, self.y_train_sample, self.test_embeddings,
                    perplexity=self.tsne_perplexity,
                    n_iter=self.tsne_n_iter,
                    random_state=self.seed
                )
        else:
            # Use classification visualization methods
            if self.use_3d_tsne:
                self.logger.info("Creating 3D classification t-SNE visualization...")
                self.train_tsne, self.test_tsne, base_fig = viz_methods['create_tsne_3d_visualization'](
                    self.train_embeddings, self.y_train_sample, self.test_embeddings,
                    perplexity=self.tsne_perplexity,
                    n_iter=self.tsne_n_iter,
                    random_state=self.seed
                )
            else:
                self.logger.info("Creating 2D classification t-SNE visualization...")
                self.train_tsne, self.test_tsne, base_fig = viz_methods['create_tsne_visualization'](
                    self.train_embeddings, self.y_train_sample, self.test_embeddings,
                    perplexity=self.tsne_perplexity,
                    n_iter=self.tsne_n_iter,
                    random_state=self.seed
                )
        
        # Close base figure to save memory
        plt.close(base_fig)
        
        # Set up class/target information based on task type
        if self.task_type == 'classification':
            # Get unique classes and set up class names
            if not hasattr(self, 'unique_classes'):
                self.unique_classes = np.unique(self.y_train_sample)
            
            # Extract semantic class names with fallback
            semantic_class_names, _ = extract_class_names_from_labels(
                labels=self.unique_classes.tolist(),
                dataset_name=kwargs.get('dataset_name', None),
                semantic_data_dir=kwargs.get('semantic_data_dir', None),
                use_semantic=self.use_semantic_names
            )
            
            # Create mapping from numeric labels to semantic names
            self.class_to_semantic = {cls: name for cls, name in zip(sorted(self.unique_classes), semantic_class_names)}
            
            # Store class names
            if class_names is not None:
                self.class_names = class_names
            else:
                self.class_names = semantic_class_names
        else:
            # For regression, we don't have class names
            self.unique_classes = None
            self.class_to_semantic = None
            self.class_names = None
        
        # Initialize multi-visualization framework if enabled
        if self.enable_multi_viz:
            # Use the reduced training data for multi-visualization
            self._initialize_multi_viz_composer(
                X_train_fit, 
                y_train_fit, 
                X_test_for_embedding if X_test is not None else None
            )
            
        self.logger.info(f"CLAM t-SNE {self.task_type} model fitted successfully")
    
    def predict(self, X_test, y_test=None, return_detailed=False, save_outputs=False, output_dir=None):
        """
        Make predictions using the fitted CLAM t-SNE classifier.
        
        Args:
            X_test: Test features (not used directly, embeddings already computed in fit)
            y_test: Test labels (for evaluation)
            return_detailed: Whether to return detailed prediction information
            save_outputs: Whether to save visualizations and outputs
            output_dir: Directory to save outputs
            
        Returns:
            predictions or detailed results dict
        """
        if self.train_tsne is None or self.test_tsne is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Load VLM
        self._load_vlm()
        
        # Import required modules
        try:
            from PIL import Image
            import io
        except ImportError as e:
            self.logger.error(f"VLM dependencies not found: {e}")
            raise ImportError("Please install required packages: pip install pillow")
        
        # Get visualization methods
        viz_methods = self._get_tsne_visualization_methods()
        
        # Parse custom viewing angles if provided
        viewing_angles = None
        if self.use_3d_tsne and 'viewing_angles' in self.modality_kwargs:
            try:
                # Parse format: "elev1,azim1;elev2,azim2;..."
                angle_pairs = self.modality_kwargs['viewing_angles'].split(';')
                viewing_angles = []
                for pair in angle_pairs:
                    elev, azim = map(int, pair.split(','))
                    viewing_angles.append((elev, azim))
                self.logger.info(f"Using custom viewing angles: {viewing_angles}")
            except Exception as e:
                self.logger.warning(f"Error parsing viewing angles: {e}. Using defaults.")
                viewing_angles = None
        
        # Make predictions for each test point
        predictions = []
        prediction_details = []
        completed_samples = 0
        
        self.logger.info(f"Starting VLM predictions for {len(self.test_tsne)} test points...")
        
        for i in range(len(self.test_tsne)):
            try:
                # Choose visualization approach based on multi-viz setting
                if self.enable_multi_viz and self.context_composer is not None:
                    # Use multi-visualization approach
                    highlight_indices = [i]  # Highlight current test point
                    
                    # Create composed visualization
                    composed_image = self.context_composer.compose_layout(
                        highlight_indices=highlight_indices,
                        layout_strategy=LayoutStrategy[self.layout_strategy.upper()]
                    )
                    
                    # Convert PIL image to format expected by VLM
                    image = composed_image
                    
                    # Create multi-visualization reasoning prompt
                    if self.task_type == 'regression':
                        # Import regression prompt functions
                        from clam.utils.vlm_prompting import create_regression_prompt
                        
                        # Generate multi-viz reasoning prompt for regression
                        reasoning_prompt = self.context_composer.generate_reasoning_prompt(
                            highlight_indices=highlight_indices,
                            custom_context=f"This is {self.modality} data with regression targets. "
                                          f"The highlighted point represents the test sample to predict.",
                            task_description=f"Predict the target value for the highlighted point based on "
                                           f"its position across multiple visualization perspectives. "
                                           f"Target range: {self.target_stats['min']:.3f} to {self.target_stats['max']:.3f}"
                        )
                        prompt = reasoning_prompt
                    else:
                        # Get visible classes across all visualizations
                        all_visible_classes = set()
                        for viz_result in self.context_composer.visualization_results:
                            if hasattr(viz_result, 'metadata') and 'classes' in viz_result.metadata:
                                all_visible_classes.update(viz_result.metadata['classes'])
                        
                        visible_classes_list = sorted(list(all_visible_classes))
                        visible_semantic_names = [self.class_to_semantic.get(cls, str(cls)) for cls in visible_classes_list]
                        
                        # Generate multi-viz reasoning prompt for classification
                        reasoning_prompt = self.context_composer.generate_reasoning_prompt(
                            highlight_indices=highlight_indices,
                            custom_context=f"This is {self.modality} data with {len(self.unique_classes)} classes: "
                                          f"{', '.join(visible_semantic_names)}. "
                                          f"The highlighted point represents the test sample to classify.",
                            task_description=f"Classify the highlighted point based on its position and "
                                           f"relationships across multiple visualization perspectives. "
                                           f"Consider cluster consistency across different methods."
                        )
                        prompt = reasoning_prompt
                        
                    legend_text = f"Multi-visualization analysis ({len(self.visualization_methods)} methods)"
                    
                else:
                    # Use legacy single visualization approach
                    # Create visualization highlighting current test point based on task type
                    if self.task_type == 'regression':
                        # Use regression visualization methods
                        if self.use_knn_connections:
                            # Create visualization with KNN connections for regression
                            if self.use_3d_tsne:
                                fig, legend_text, metadata = viz_methods['create_regression_tsne_3d_plot_with_knn'](
                                self.train_tsne, self.test_tsne, self.y_train_sample,
                                self.train_embeddings, self.test_embeddings,
                                highlight_test_idx=i,
                                k=self.knn_k,
                                figsize=(12, 9),
                                viewing_angles=viewing_angles,
                                zoom_factor=self.tsne_zoom_factor
                            )
                            else:
                                fig, legend_text, metadata = viz_methods['create_regression_tsne_plot_with_knn'](
                                self.train_tsne, self.test_tsne, self.y_train_sample,
                                self.train_embeddings, self.test_embeddings,
                                highlight_test_idx=i,
                                k=self.knn_k,
                                figsize=(10, 8),
                                zoom_factor=self.tsne_zoom_factor
                            )
                        else:
                            # Create standard regression visualization
                            if self.use_3d_tsne:
                                fig, legend_text, metadata = viz_methods['create_combined_regression_tsne_3d_plot'](
                                self.train_tsne, self.test_tsne, self.y_train_sample,
                                highlight_test_idx=i,
                                figsize=(12, 9),
                                viewing_angles=viewing_angles,
                                zoom_factor=self.tsne_zoom_factor
                            )
                            else:
                                fig, legend_text, metadata = viz_methods['create_combined_regression_tsne_plot'](
                                self.train_tsne, self.test_tsne, self.y_train_sample,
                                highlight_test_idx=i,
                                figsize=(8, 6),
                                zoom_factor=self.tsne_zoom_factor
                            )
                    else:
                        # Use classification visualization methods
                        if self.use_knn_connections:
                            # Create visualization with KNN connections
                            if self.use_3d_tsne:
                                fig, legend_text, metadata = viz_methods['create_tsne_3d_plot_with_knn'](
                                self.train_tsne, self.test_tsne, self.y_train_sample,
                                self.train_embeddings, self.test_embeddings,
                                highlight_test_idx=i,
                                k=self.knn_k,
                                figsize=(12, 9),
                                viewing_angles=viewing_angles,
                                zoom_factor=self.tsne_zoom_factor
                            )
                            else:
                                fig, legend_text, metadata = viz_methods['create_tsne_plot_with_knn'](
                                self.train_tsne, self.test_tsne, self.y_train_sample,
                                self.train_embeddings, self.test_embeddings,
                                highlight_test_idx=i,
                                k=self.knn_k,
                                figsize=(10, 8),
                                zoom_factor=self.tsne_zoom_factor
                            )
                        else:
                            # Create standard visualization
                            if self.use_3d_tsne:
                                fig, legend_text, metadata = viz_methods['create_combined_tsne_3d_plot'](
                                self.train_tsne, self.test_tsne, self.y_train_sample,
                                highlight_test_idx=i,
                                figsize=(12, 9),
                                viewing_angles=viewing_angles,
                                zoom_factor=self.tsne_zoom_factor
                            )
                            else:
                                fig, legend_text, metadata = viz_methods['create_combined_tsne_plot'](
                                self.train_tsne, self.test_tsne, self.y_train_sample,
                                highlight_test_idx=i,
                                figsize=(8, 6),
                                zoom_factor=self.tsne_zoom_factor
                            )
                
                # Convert plot to image (only for legacy single visualization)
                if not self.enable_multi_viz or self.context_composer is None:
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='png', dpi=self.image_dpi, bbox_inches='tight', facecolor='white')
                    img_buffer.seek(0)
                    image = Image.open(img_buffer)
                    plt.close(fig)
                
                # Convert to RGB if needed
                if self.force_rgb_mode and image.mode != 'RGB':
                    if image.mode == 'RGBA':
                        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                        rgb_image.paste(image, mask=image.split()[3] if len(image.split()) == 4 else None)
                        image = rgb_image
                    else:
                        image = image.convert('RGB')
                
                # Resize if needed
                if image.width > self.max_vlm_image_size or image.height > self.max_vlm_image_size:
                    ratio = min(self.max_vlm_image_size / image.width, self.max_vlm_image_size / image.height)
                    new_width = int(image.width * ratio)
                    new_height = int(image.height * ratio)
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Create prompt based on task type and visualization approach
                if self.enable_multi_viz and self.context_composer is not None:
                    # Multi-viz prompt already created above
                    pass
                else:
                    # Create legacy single visualization prompt
                    if self.task_type == 'regression':
                        # Import regression prompt functions
                        from clam.utils.vlm_prompting import create_regression_prompt, parse_vlm_response
                        
                        # Create regression prompt
                        prompt = create_regression_prompt(
                            target_stats=self.target_stats,
                            modality=self.modality,
                            use_knn=self.use_knn_connections,
                            use_3d=self.use_3d_tsne,
                            knn_k=self.knn_k if self.use_knn_connections else None,
                            legend_text=legend_text,
                            dataset_description=f"{self.modality.title()} data embedded using appropriate features"
                        )
                        
                        # Create conversation
                        conversation = create_vlm_conversation(image, prompt)
                        
                        # Generate response with API-aware config for regression
                        gen_config = GenerationConfig(
                            max_new_tokens=16384,  # Generous limit for thinking + regression
                            temperature=0.1,
                            do_sample=True,
                            enable_thinking=self.enable_thinking and self.is_api_model,
                            thinking_summary=False
                        )
                        
                        response = self.vlm_wrapper.generate_from_conversation(conversation, gen_config)
                        
                        # Parse prediction for regression
                        prediction = parse_vlm_response(
                            response, 
                            unique_classes=None, 
                            logger_instance=self.logger, 
                            use_semantic_names=False,
                            task_type='regression',
                            target_stats=self.target_stats
                        )
                    else:
                        # Classification logic
                        # Get visible classes
                        visible_classes = set(metadata.get('classes', []))
                        if metadata.get('knn_info') and 'neighbor_classes' in metadata['knn_info']:
                            visible_classes.update(set(metadata['knn_info']['neighbor_classes']))
                        
                        visible_classes_list = sorted(list(visible_classes))
                        visible_semantic_names = [self.class_to_semantic[cls] for cls in visible_classes_list]
                        
                        # Create classification prompt
                        from clam.utils.vlm_prompting import create_classification_prompt, parse_vlm_response
                        prompt = create_classification_prompt(
                            class_names=visible_semantic_names,
                            modality=self.modality,
                            use_knn=self.use_knn_connections,
                            use_3d=self.use_3d_tsne,
                            knn_k=self.knn_k if self.use_knn_connections else None,
                            legend_text=legend_text,
                            dataset_description=f"{self.modality.title()} data embedded using appropriate features",
                            use_semantic_names=self.use_semantic_names
                        )
                        
                        # Create conversation
                        conversation = create_vlm_conversation(image, prompt)
                        
                        # Generate response with API-aware config for classification
                        gen_config = GenerationConfig(
                            max_new_tokens=16384,  # Generous limit for thinking + classification
                            temperature=0.1,
                            do_sample=True,
                            enable_thinking=self.enable_thinking and self.is_api_model,
                            thinking_summary=False
                        )
                        
                        response = self.vlm_wrapper.generate_from_conversation(conversation, gen_config)
                        
                        # Parse prediction for classification
                        prediction = parse_vlm_response(
                            response, 
                            visible_semantic_names, 
                            self.logger, 
                            use_semantic_names=True,
                            task_type='classification'
                        )
                        
                        # Map back to numeric label if needed
                        if prediction in visible_semantic_names:
                            semantic_to_numeric = {name: cls for cls, name in self.class_to_semantic.items() if cls in visible_classes_list}
                            prediction = semantic_to_numeric.get(prediction, prediction)
                
                # For multi-visualization, handle VLM response generation
                if self.enable_multi_viz and self.context_composer is not None:
                    # Create conversation and generate response for multi-viz
                    conversation = create_vlm_conversation(image, prompt)
                    
                    gen_config = GenerationConfig(
                        max_new_tokens=16384,
                        temperature=0.1,
                        do_sample=True,
                        enable_thinking=self.enable_thinking and self.is_api_model,
                        thinking_summary=False
                    )
                    
                    response = self.vlm_wrapper.generate_from_conversation(conversation, gen_config)
                    
                    # Parse response based on task type
                    from clam.utils.vlm_prompting import parse_vlm_response
                    
                    if self.task_type == 'classification':
                        # Get all visible classes across visualizations
                        all_visible_classes = set()
                        for viz_result in self.context_composer.visualization_results:
                            if hasattr(viz_result, 'metadata') and 'classes' in viz_result.metadata:
                                all_visible_classes.update(viz_result.metadata['classes'])
                        
                        visible_classes_list = sorted(list(all_visible_classes))
                        visible_semantic_names = [self.class_to_semantic.get(cls, str(cls)) for cls in visible_classes_list]
                        
                        prediction = parse_vlm_response(
                            response, 
                            visible_semantic_names, 
                            self.logger, 
                            use_semantic_names=True,
                            task_type='classification'
                        )
                        
                        # Map back to numeric label if needed
                        if prediction in visible_semantic_names:
                            semantic_to_numeric = {name: cls for cls, name in self.class_to_semantic.items() if cls in visible_classes_list}
                            prediction = semantic_to_numeric.get(prediction, prediction)
                    else:
                        # Regression
                        prediction = parse_vlm_response(
                            response, 
                            unique_classes=None, 
                            logger_instance=self.logger, 
                            use_semantic_names=False,
                            task_type='regression',
                            target_stats=self.target_stats
                        )
                
                predictions.append(prediction)
                
                # Store details
                if return_detailed and y_test is not None:
                    true_label = y_test[i] if hasattr(y_test, '__getitem__') else y_test.iloc[i]
                    prediction_details.append({
                        'test_point_idx': i,
                        'vlm_response': response,
                        'parsed_prediction': prediction,
                        'true_label': true_label,
                        'tsne_coords': self.test_tsne[i].tolist(),
                        'image_size': f"{image.width}x{image.height}",
                        'visible_classes': visible_classes_list
                    })
                
                completed_samples = i + 1
                
                # Log progress
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Completed {i + 1}/{len(self.test_tsne)} predictions")
                
            except Exception as e:
                self.logger.error(f"VLM prediction failed for test point {i}: {e}")
                # Use random prediction as fallback
                prediction = np.random.choice(self.unique_classes)
                predictions.append(prediction)
                completed_samples = i + 1
        
        if return_detailed:
            return {
                'predictions': predictions,
                'prediction_details': prediction_details,
                'completed_samples': completed_samples,
                'completion_rate': completed_samples / len(self.test_tsne) if len(self.test_tsne) > 0 else 0.0
            }
        else:
            return predictions
    
    def evaluate(self, X_test, y_test, return_detailed=False, save_outputs=False, output_dir=None):
        """
        Evaluate the classifier on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            return_detailed: Whether to return detailed results
            save_outputs: Whether to save outputs
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with evaluation metrics
        """
        start_time = time.time()
        
        # Make predictions
        detailed_results = self.predict(X_test, y_test, return_detailed=True, save_outputs=save_outputs, output_dir=output_dir)
        predictions = detailed_results['predictions']
        completed_samples = detailed_results['completed_samples']
        
        # Calculate metrics
        if completed_samples > 0:
            # Get partial ground truth
            y_test_partial = y_test[:completed_samples] if hasattr(y_test, '__getitem__') else list(y_test)[:completed_samples]
            
            # Convert predictions to same type as ground truth
            predictions_converted = []
            target_type = type(y_test_partial[0])
            
            for pred in predictions:
                try:
                    predictions_converted.append(target_type(pred))
                except (ValueError, TypeError):
                    predictions_converted.append(pred)
            
            # Calculate metrics using shared utility
            from clam.utils.llm_evaluation_utils import calculate_llm_metrics
            metrics = calculate_llm_metrics(
                y_test_partial, predictions_converted, self.unique_classes,
                all_class_log_probs=None, logger=self.logger
            )
        else:
            metrics = {
                'accuracy': 0.0,
                'balanced_accuracy': 0.0,
                'roc_auc': None,
                'f1_macro': None,
                'f1_micro': None,
                'f1_weighted': None,
                'precision_macro': None,
                'recall_macro': None
            }
        
        # Calculate timing
        total_time = time.time() - start_time
        
        # Build results
        results = {
            'model_name': f'CLAM-t-SNE-{self.modality}',
            'accuracy': float(metrics['accuracy']),
            'balanced_accuracy': float(metrics['balanced_accuracy']),
            'prediction_time': float(total_time),
            'total_time': float(total_time),
            'num_test_samples': len(X_test) if hasattr(X_test, '__len__') else len(self.test_tsne),
            'completed_samples': completed_samples,
            'completion_rate': detailed_results['completion_rate'],
            'num_classes': len(self.unique_classes),
            'predictions': predictions_converted if 'predictions_converted' in locals() else predictions,
            'ground_truth': (y_test[:completed_samples].tolist() if hasattr(y_test[:completed_samples], 'tolist')
                           else list(y_test)[:completed_samples]) if completed_samples > 0 else [],
            # Additional metrics
            'roc_auc': float(metrics['roc_auc']) if metrics['roc_auc'] is not None else None,
            'f1_macro': float(metrics['f1_macro']) if metrics['f1_macro'] is not None else None,
            'f1_micro': float(metrics['f1_micro']) if metrics['f1_micro'] is not None else None,
            'f1_weighted': float(metrics['f1_weighted']) if metrics['f1_weighted'] is not None else None,
            'precision_macro': float(metrics['precision_macro']) if metrics['precision_macro'] is not None else None,
            'recall_macro': float(metrics['recall_macro']) if metrics['recall_macro'] is not None else None,
            # Model configuration
            'config': self.get_config()
        }
        
        if return_detailed:
            results.update({
                'prediction_details': detailed_results.get('prediction_details', [])
            })
        
        return results
    
    def get_config(self):
        """Get configuration dictionary."""
        return {
            'modality': self.modality,
            'vlm_model_id': self.vlm_model_id,
            'embedding_size': self.embedding_size,
            'tsne_perplexity': self.tsne_perplexity,
            'tsne_n_iter': self.tsne_n_iter,
            'use_3d_tsne': self.use_3d_tsne,
            'use_knn_connections': self.use_knn_connections,
            'knn_k': self.knn_k,
            'max_vlm_image_size': self.max_vlm_image_size,
            'image_dpi': self.image_dpi,
            'force_rgb_mode': self.force_rgb_mode,
            'tsne_zoom_factor': self.tsne_zoom_factor,
            'max_tabpfn_samples': self.max_tabpfn_samples,
            'use_semantic_names': self.use_semantic_names,
            'device': self.device,
            'backend': self.backend,
            'enable_thinking': self.enable_thinking,
            'openai_model': self.openai_model,
            'gemini_model': self.gemini_model,
            'api_model': self.api_model,
            'effective_model_id': self.effective_model_id,
            'is_api_model': self.is_api_model,
            'seed': self.seed,
            # Multi-visualization parameters
            'enable_multi_viz': self.enable_multi_viz,
            'visualization_methods': self.visualization_methods,
            'layout_strategy': self.layout_strategy,
            'reasoning_focus': self.reasoning_focus,
            'multi_viz_config': self.multi_viz_config
        }


def evaluate_clam_tsne(dataset, args):
    """
    Evaluate CLAM t-SNE baseline on a dataset (legacy function for backward compatibility).
    
    This function maintains compatibility with existing tabular LLM baseline scripts.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating CLAM t-SNE on dataset {dataset['name']}")
    
    try:
        # Import required utilities
        from clam.utils import (
            drop_feature_for_oom,
            is_oom_error,
            apply_feature_reduction
        )
        
        # Split data
        X, y = dataset["X"], dataset["y"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed
        )
        
        # Limit test samples if specified
        if args.max_test_samples and args.max_test_samples < len(X_test):
            X_test = X_test[:args.max_test_samples]
            y_test = y_test[:args.max_test_samples]
        
        # Create classifier
        classifier = ClamTsneClassifier(
            modality="tabular",
            vlm_model_id=getattr(args, 'vlm_model_id', "Qwen/Qwen2.5-VL-32B-Instruct"),
            embedding_size=getattr(args, 'embedding_size', 1000),
            tsne_perplexity=getattr(args, 'tsne_perplexity', 30),
            tsne_n_iter=getattr(args, 'tsne_n_iter', 1000),
            use_3d_tsne=getattr(args, 'use_3d_tsne', False),
            use_knn_connections=getattr(args, 'use_knn_connections', False),
            knn_k=getattr(args, 'knn_k', 5),
            max_vlm_image_size=getattr(args, 'max_vlm_image_size', 2048),
            image_dpi=getattr(args, 'image_dpi', 100),
            force_rgb_mode=getattr(args, 'force_rgb_mode', True),
            tsne_zoom_factor=getattr(args, 'tsne_zoom_factor', 2.0),
            max_tabpfn_samples=getattr(args, 'max_tabpfn_samples', 3000),
            cache_dir=getattr(args, 'cache_dir', None),
            use_semantic_names=getattr(args, 'use_semantic_names', False),
            device=args.device,
            backend=getattr(args, 'backend', 'auto'),
            tensor_parallel_size=getattr(args, 'tensor_parallel_size', 1),
            gpu_memory_utilization=getattr(args, 'gpu_memory_utilization', 0.9),
            enable_thinking=getattr(args, 'enable_thinking', True),
            openai_model=getattr(args, 'openai_model', None),
            gemini_model=getattr(args, 'gemini_model', None),
            api_model=getattr(args, 'api_model', None),
            seed=args.seed,
            # Multi-visualization parameters
            enable_multi_viz=getattr(args, 'enable_multi_viz', False),
            visualization_methods=getattr(args, 'visualization_methods', ['tsne']),
            layout_strategy=getattr(args, 'layout_strategy', 'adaptive_grid'),
            reasoning_focus=getattr(args, 'reasoning_focus', 'classification'),
            multi_viz_config=getattr(args, 'multi_viz_config', {}),
            # Pass additional args as kwargs
            viewing_angles=getattr(args, 'viewing_angles', None),
            feature_selection_threshold=getattr(args, 'feature_selection_threshold', 500)
        )
        
        # Fit and evaluate
        classifier.fit(X_train, y_train, X_test, dataset_name=dataset['name'])
        results = classifier.evaluate(
            X_test, y_test, 
            return_detailed=True,
            save_outputs=getattr(args, 'save_sample_visualizations', True),
            output_dir=getattr(args, 'output_dir', None)
        )
        
        # Add dataset information
        results.update({
            'dataset_name': dataset['name'],
            'dataset_id': dataset['id'],
            'task_id': dataset['id']
        })
        
        logger.info(f"CLAM t-SNE accuracy on {dataset['name']}: {results['accuracy']:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating CLAM t-SNE: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'model_name': 'CLAM-t-SNE',
            'dataset_name': dataset['name'],
            'error': str(e)
        }