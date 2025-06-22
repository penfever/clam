"""
CLAM t-SNE baseline for image classification.

This implements the proper CLAM pipeline for images:
DINOV2 embeddings → t-SNE visualization → VLM classification

Based on the centralized implementation in clam.models.clam_tsne
"""

import os
import numpy as np
import torch
import time
import logging
import tempfile
import json
import io
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
from PIL import Image

# Import CLAM utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from clam.utils.vlm_prompting import create_classification_prompt, parse_vlm_response, create_vlm_conversation
from clam.viz.utils.common import (
    plot_to_image, save_visualization_with_metadata, create_output_directories,
    generate_visualization_filename, close_figure_safely
)
from clam.utils.platform_utils import (
    get_optimal_device, get_platform_compatible_dtype, configure_model_kwargs_for_platform,
    is_mac_platform, log_platform_info
)
from clam.utils.json_utils import convert_for_json_serialization

from clam.data.embeddings import get_dinov2_embeddings
from clam.viz.tsne_functions import (
    create_tsne_visualization,
    create_tsne_3d_visualization,
    create_combined_tsne_plot,
    create_combined_tsne_3d_plot,
    create_tsne_plot_with_knn,
    create_tsne_3d_plot_with_knn
)

logger = logging.getLogger(__name__)


class ClamImageTsneClassifier:
    """
    CLAM t-SNE classifier for image classification.
    
    Pipeline: DINOV2 embeddings → t-SNE visualization → VLM classification
    """
    
    def __init__(
        self,
        dinov2_model: str = "dinov2_vitb14",
        embedding_size: int = 1000,
        tsne_perplexity: float = 30.0,
        tsne_n_iter: int = 1000,
        vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        use_3d_tsne: bool = False,
        use_knn_connections: bool = False,
        knn_k: int = 5,
        max_vlm_image_size: int = 1024,
        image_dpi: int = 100,
        force_rgb_mode: bool = True,
        zoom_factor: float = 4.0,
        use_pca_backend: bool = False,
        max_train_plot_samples: int = 1000,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_semantic_names: bool = False,
        enable_thinking: bool = True,
        openai_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
        api_model: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize CLAM image t-SNE classifier.
        
        Args:
            dinov2_model: DINOV2 model variant to use
            embedding_size: Target embedding size
            tsne_perplexity: t-SNE perplexity parameter
            tsne_n_iter: Number of t-SNE iterations
            vlm_model_id: Vision Language Model ID
            use_3d_tsne: Whether to use 3D t-SNE
            use_knn_connections: Whether to show KNN connections
            knn_k: Number of nearest neighbors to show
            max_vlm_image_size: Maximum image size for VLM
            image_dpi: DPI for saving visualizations
            force_rgb_mode: Convert images to RGB mode
            zoom_factor: Zoom factor for t-SNE visualizations
            use_pca_backend: Use PCA instead of t-SNE for dimensionality reduction
            max_train_plot_samples: Maximum training samples to include in plots
            cache_dir: Directory for caching embeddings
            device: Device for model inference
            seed: Random seed
        """
        self.dinov2_model = dinov2_model
        self.embedding_size = embedding_size
        self.tsne_perplexity = tsne_perplexity
        self.tsne_n_iter = tsne_n_iter
        self.vlm_model_id = vlm_model_id
        self.use_3d = use_3d_tsne
        self.use_knn_connections = use_knn_connections
        self.knn_k = knn_k
        self.max_vlm_image_size = max_vlm_image_size
        self.image_dpi = image_dpi
        self.force_rgb_mode = force_rgb_mode
        self.zoom_factor = zoom_factor
        self.use_pca_backend = use_pca_backend
        self.max_train_plot_samples = max_train_plot_samples
        self.cache_dir = cache_dir
        self.device = device or get_optimal_device(force_cpu=is_mac_platform())
        self.use_semantic_names = use_semantic_names
        self.enable_thinking = enable_thinking
        self.openai_model = openai_model
        self.gemini_model = gemini_model
        self.api_model = api_model
        self.seed = seed
        
        # Determine the effective model to use (API models take precedence)
        self.effective_model_id = self._determine_effective_model()
        self.is_api_model = self._is_api_model(self.effective_model_id)
        
        # State variables
        self.is_fitted = False
        self.train_embeddings = None
        self.train_embeddings_plot = None
        self.test_embeddings = None
        self.train_tsne = None
        self.test_tsne = None
        self.y_train = None
        self.y_train_plot = None
        self.unique_classes = None
        self.vlm_wrapper = None
        self.save_every_n = 1  # Default: save every prediction
    
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
        
    def fit(
        self,
        train_image_paths: List[str],
        train_labels: List[int],
        test_image_paths: List[str],
        class_names: Optional[List[str]] = None
    ) -> 'ClamImageTsneClassifier':
        """
        Fit the CLAM image t-SNE classifier.
        
        Args:
            train_image_paths: List of training image paths
            train_labels: List of training labels
            test_image_paths: List of test image paths
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting CLAM image t-SNE classifier on {len(train_image_paths)} training images")
        start_time = time.time()
        
        # Store training labels and class names
        self.y_train = np.array(train_labels)
        self.unique_classes = np.unique(self.y_train)
        self.class_names = class_names
        
        # Extract DINOV2 embeddings for training and test sets
        logger.info("Extracting DINOV2 embeddings for training set...")
        self.train_embeddings = get_dinov2_embeddings(
            train_image_paths,
            model_name=self.dinov2_model,
            embedding_size=self.embedding_size,
            cache_dir=self.cache_dir,
            dataset_name="train",
            device=self.device
        )
        
        # Limit training samples for plotting if needed
        if len(self.train_embeddings) > self.max_train_plot_samples:
            logger.info(f"Limiting training samples for plotting: {len(self.train_embeddings)} -> {self.max_train_plot_samples}")
            # Use stratified sampling to maintain class balance
            from sklearn.model_selection import train_test_split
            _, plot_indices = train_test_split(
                np.arange(len(self.train_embeddings)),
                train_size=self.max_train_plot_samples,
                random_state=self.seed,
                stratify=self.y_train
            )
            
            self.train_embeddings_plot = self.train_embeddings[plot_indices]
            self.y_train_plot = self.y_train[plot_indices]
        else:
            self.train_embeddings_plot = self.train_embeddings
            self.y_train_plot = self.y_train
        
        logger.info("Extracting DINOV2 embeddings for test set...")
        self.test_embeddings = get_dinov2_embeddings(
            test_image_paths,
            model_name=self.dinov2_model,
            embedding_size=self.embedding_size,
            cache_dir=self.cache_dir,
            dataset_name="test",
            device=self.device
        )
        
        # Create dimensionality reduction visualization
        if self.use_pca_backend:
            logger.info("Creating PCA visualization...")
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            import warnings
            
            # Combine embeddings for joint processing
            combined_embeddings = np.vstack([self.train_embeddings_plot, self.test_embeddings])
            n_train_plot = len(self.train_embeddings_plot)
            
            # Standardize embeddings to prevent numerical instability
            logger.info("Standardizing embeddings for PCA...")
            scaler = StandardScaler()
            combined_embeddings_scaled = scaler.fit_transform(combined_embeddings)
            
            # Apply PCA to standardized data
            n_components = 3 if self.use_3d_tsne else 2
            pca = PCA(n_components=n_components, random_state=self.seed)
            pca_results = pca.fit_transform(combined_embeddings_scaled)
            
            # Split back into train and test
            self.train_tsne = pca_results[:n_train_plot]
            self.test_tsne = pca_results[n_train_plot:]
            
            logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
            base_fig = None  # No initial figure for PCA
        else:
            # Use t-SNE with standardized embeddings
            from sklearn.preprocessing import StandardScaler
            
            # Combine embeddings for joint standardization
            combined_embeddings = np.vstack([self.train_embeddings_plot, self.test_embeddings])
            n_train_plot = len(self.train_embeddings_plot)
            
            # Standardize embeddings for better t-SNE stability
            logger.info("Standardizing embeddings for t-SNE...")
            scaler = StandardScaler()
            combined_embeddings_scaled = scaler.fit_transform(combined_embeddings)
            
            # Split back into train and test
            train_embeddings_scaled = combined_embeddings_scaled[:n_train_plot]
            test_embeddings_scaled = combined_embeddings_scaled[n_train_plot:]
            
            if self.use_3d_tsne:
                logger.info("Creating 3D t-SNE visualization...")
                self.train_tsne, self.test_tsne, base_fig = create_tsne_3d_visualization(
                    train_embeddings_scaled, self.y_train_plot, test_embeddings_scaled,
                    perplexity=self.tsne_perplexity,
                    n_iter=self.tsne_n_iter,
                    random_state=self.seed,
                    class_names=None,  # Will be handled in combined plots
                    use_semantic_names=self.use_semantic_names
                )
            else:
                logger.info("Creating 2D t-SNE visualization...")
                self.train_tsne, self.test_tsne, base_fig = create_tsne_visualization(
                    train_embeddings_scaled, self.y_train_plot, test_embeddings_scaled,
                    perplexity=self.tsne_perplexity,
                    n_iter=self.tsne_n_iter,
                    random_state=self.seed,
                    class_names=None,  # Will be handled in combined plots
                    use_semantic_names=self.use_semantic_names
                )
        
        # Close the base figure to save memory
        if base_fig is not None:
            plt.close(base_fig)
        
        # Load VLM model
        logger.info(f"Loading Vision Language Model: {self.vlm_model_id}")
        self.vlm_wrapper = self._load_vlm_model()
        
        self.is_fitted = True
        elapsed_time = time.time() - start_time
        logger.info(f"CLAM image t-SNE classifier fitted in {elapsed_time:.2f} seconds")
        
        return self
    
    def _load_vlm_model(self):
        """Load the Vision Language Model (local or API-based) using standardized model loader."""
        try:
            # Use the centralized model loader from CLAM
            from clam.utils.model_loader import model_loader
            
            model_to_load = self.effective_model_id
            logger.info(f"Loading Vision Language Model: {model_to_load}")
            
            if self.is_api_model:
                # API model - minimal configuration needed
                logger.info("Using API-based VLM (OpenAI/Gemini)")
                vlm_kwargs = {}
                device = None  # API models don't need device specification
                backend = 'auto'  # Auto-detect API provider
            else:
                # Local model - get platform-compatible kwargs
                logger.info("Using local VLM")
                vlm_kwargs = configure_model_kwargs_for_platform(
                    device=self.device,
                    torch_dtype=get_platform_compatible_dtype(self.device)
                )
                device = self.device
                backend = 'auto'
            
            # Load VLM using centralized model loader
            vlm_wrapper = model_loader.load_vlm(
                model_to_load, 
                backend=backend,
                device=device, 
                tensor_parallel_size=1 if not self.is_api_model else None,
                gpu_memory_utilization=0.9 if not self.is_api_model else None,
                **vlm_kwargs
            )
            
            return vlm_wrapper
            
        except ImportError:
            logger.warning("CLAM model loader not available, using simple VLM implementation")
            return self._create_simple_vlm()
    
    def _create_simple_vlm(self):
        """Create a simple VLM wrapper for testing."""
        class SimpleVLMWrapper:
            def __init__(self, model_id, device):
                self.model_id = model_id
                self.device = device
                self.model = None
                self.processor = None
                self._load_model()
            
            def _load_model(self):
                try:
                    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                    import sys
                    import warnings
                    
                    logger.info(f"Loading simple VLM: {self.model_id}")
                    
                    # Get platform-compatible configuration
                    model_kwargs = configure_model_kwargs_for_platform(device=self.device)
                    
                    # Suppress deprecation warnings during model loading
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        warnings.filterwarnings("ignore", category=UserWarning)
                        
                        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                            self.model_id,
                            **model_kwargs
                        )
                        
                        self.processor = AutoProcessor.from_pretrained(
                            self.model_id,
                            trust_remote_code=True
                        )
                    
                except ImportError as e:
                    logger.error(f"VLM dependencies not available: {e}")
                    self.model = "mock"
                    self.processor = "mock"
                except Exception as e:
                    logger.error(f"Failed to load VLM: {e}")
                    self.model = "mock"
                    self.processor = "mock"
            
            def generate_from_conversation(self, conversation, gen_config):
                if self.model == "mock":
                    # Mock response for testing
                    import random
                    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # CIFAR-10 classes
                    pred = random.choice(classes)
                    return f"Class: {pred} | Reasoning: Random prediction for testing"
                
                # Extract image and text from conversation
                image = None
                text = ""
                for content in conversation[0]["content"]:
                    if content["type"] == "image":
                        image = content["image"]
                    elif content["type"] == "text":
                        text = content["text"]
                
                # Process inputs
                inputs = self.processor(
                    text=text,
                    images=image,
                    return_tensors="pt"
                )
                
                # Generate response
                # Note: temperature is only effective when do_sample=True
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=gen_config.max_new_tokens,
                        temperature=gen_config.temperature,
                        do_sample=gen_config.do_sample
                    )
                
                # Decode response
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
                
                # Extract just the generated part (remove input prompt)
                prompt_length = len(self.processor.decode(inputs['input_ids'][0], skip_special_tokens=True))
                generated_text = response[prompt_length:].strip()
                
                return generated_text
        
        return SimpleVLMWrapper(self.vlm_model_id, self.device)
    
    def predict(self, test_image_paths: List[str], save_outputs: bool = False, output_dir: Optional[str] = None) -> Tuple[List[Any], List[Dict]]:
        """
        Predict labels for test images using VLM on t-SNE visualizations.
        
        Args:
            test_image_paths: List of test image paths
            save_outputs: Whether to save visualizations and responses
            output_dir: Directory to save outputs (required if save_outputs=True)
            
        Returns:
            Tuple of (predictions, detailed_outputs)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        logger.info(f"Predicting labels for {len(test_image_paths)} test images using VLM")
        
        predictions = []
        detailed_outputs = []
        if self.unique_classes is None:
            self.unique_classes = []
        class_list_str = ", ".join([str(cls) for cls in self.unique_classes])
        
        # Setup output directories if saving
        if save_outputs and output_dir:
            dir_paths = create_output_directories(output_dir, ['image_visualizations'])
            viz_dir = dir_paths['image_visualizations']
        
        for i in range(len(self.test_tsne)):
            try:
                # Create visualization highlighting current test point
                if self.use_knn_connections and not self.use_pca_backend:
                    # Create visualization with KNN connections (only for t-SNE)
                    if self.use_3d_tsne:
                        fig, legend_text, metadata = create_tsne_3d_plot_with_knn(
                            self.train_tsne, self.test_tsne, self.y_train_plot,
                            self.train_embeddings_plot, self.test_embeddings,
                            highlight_test_idx=i,
                            k=self.knn_k,
                            figsize=(12, 9),
                            zoom_factor=self.zoom_factor,
                            class_names=self.class_names,
                            use_semantic_names=self.use_semantic_names
                        )
                    else:
                        fig, legend_text, metadata = create_tsne_plot_with_knn(
                            self.train_tsne, self.test_tsne, self.y_train_plot,
                            self.train_embeddings_plot, self.test_embeddings,
                            highlight_test_idx=i,
                            k=self.knn_k,
                            figsize=(10, 8),
                            zoom_factor=self.zoom_factor,
                            class_names=self.class_names,
                            use_semantic_names=self.use_semantic_names
                        )
                else:
                    # Create standard visualization
                    if self.use_3d_tsne:
                        fig, legend_text, metadata = create_combined_tsne_3d_plot(
                            self.train_tsne, self.test_tsne, self.y_train_plot, 
                            highlight_test_idx=i,
                            figsize=(12, 9),
                            zoom_factor=self.zoom_factor,
                            class_names=self.class_names,
                            use_semantic_names=self.use_semantic_names
                        )
                    else:
                        fig, legend_text, metadata = create_combined_tsne_plot(
                            self.train_tsne, self.test_tsne, self.y_train_plot, 
                            highlight_test_idx=i,
                            figsize=(8, 6),
                            zoom_factor=self.zoom_factor,
                            class_names=self.class_names,
                            use_semantic_names=self.use_semantic_names
                        )
                
                # Convert plot to image using utility
                image = plot_to_image(
                    fig, 
                    dpi=self.image_dpi,
                    force_rgb=self.force_rgb_mode,
                    max_size=self.max_vlm_image_size
                )
                
                # Create prompt using utility function
                prompt = create_classification_prompt(
                    class_names=self.class_names if self.use_semantic_names else self.unique_classes.tolist(),
                    modality="image",
                    use_knn=self.use_knn_connections and not self.use_pca_backend,
                    use_3d=self.use_3d_tsne,
                    knn_k=self.knn_k if self.use_knn_connections else None,
                    legend_text=legend_text,
                    dataset_description="Image data embedded using DINOV2 features",
                    use_semantic_names=self.use_semantic_names
                )

                # Create conversation using utility
                conversation = create_vlm_conversation(image, prompt)
                
                # Import the proper GenerationConfig class
                try:
                    from clam.utils.model_loader import GenerationConfig
                    gen_config = GenerationConfig(
                        max_new_tokens=16384,  # Generous limit for thinking + classification
                        temperature=0.1,
                        do_sample=True,
                        enable_thinking=self.enable_thinking and self.is_api_model,
                        thinking_summary=False
                    )
                except ImportError:
                    # Fallback to simple config if import fails
                    class SimpleGenConfig:
                        def __init__(self, **kwargs):
                            for k, v in kwargs.items():
                                setattr(self, k, v)
                        
                        def to_transformers_kwargs(self):
                            return {
                                'max_new_tokens': getattr(self, 'max_new_tokens', 100),
                                'temperature': getattr(self, 'temperature', 0.1),
                                'do_sample': getattr(self, 'do_sample', True)
                            }
                    
                    gen_config = SimpleGenConfig(
                        max_new_tokens=16384,  # Use higher limit even for fallback
                        temperature=0.1,
                        do_sample=True
                    )
                
                # Generate response
                response = self.vlm_wrapper.generate_from_conversation(conversation, gen_config)
                
                # Parse prediction using utility
                prediction = parse_vlm_response(response, self.unique_classes, logger, self.use_semantic_names)
                predictions.append(prediction)
                
                # Save visualization and capture detailed output (reduced frequency)
                should_save_viz = save_outputs and output_dir and (i % self.save_every_n == 0 or i == 0 or i == len(self.test_tsne) - 1)
                if should_save_viz:
                    # Generate standardized filename
                    backend_name = "pca" if self.use_pca_backend else "tsne"
                    viz_filename = generate_visualization_filename(
                        sample_index=i,
                        backend=backend_name,
                        dimensions='3d' if self.use_3d_tsne else '2d',
                        use_knn=self.use_knn_connections and not self.use_pca_backend,
                        knn_k=self.knn_k if self.use_knn_connections else None
                    )
                    
                    viz_path = os.path.join(viz_dir, viz_filename)
                    
                    # Save using utility function
                    save_info = save_visualization_with_metadata(
                        fig, viz_path, 
                        metadata=convert_for_json_serialization({
                            'sample_index': i,
                            'backend': backend_name,
                            'use_3d': self.use_3d_tsne,
                            'use_knn': self.use_knn_connections and not self.use_pca_backend,
                            'prediction': prediction,
                            'vlm_model': self.vlm_model_id
                        }),
                        dpi=self.image_dpi
                    )
                
                # Store detailed output
                detailed_outputs.append({
                    'sample_index': i,
                    'test_point_coords': self.test_tsne[i].tolist(),
                    'image_path': test_image_paths[i] if i < len(test_image_paths) else None,
                    'image_size': f"{image.width}x{image.height}",
                    'image_mode': image.mode,
                    'prompt': prompt,
                    'vlm_model': self.vlm_model_id,
                    'vlm_response': response,
                    'parsed_prediction': prediction,
                    'visualization_saved': should_save_viz,
                    'visualization_path': viz_path if should_save_viz else None,
                    'backend_params': {
                        'use_pca_backend': self.use_pca_backend,
                        'use_3d_tsne': self.use_3d_tsne,
                        'use_knn_connections': self.use_knn_connections,
                        'knn_k': self.knn_k if self.use_knn_connections else None,
                        'zoom_factor': self.zoom_factor,
                        'max_train_plot_samples': self.max_train_plot_samples
                    }
                })
                
                close_figure_safely(fig)
                
                # Log progress less frequently
                if (i + 1) % 50 == 0 or (i + 1) == len(self.test_tsne):
                    logger.info(f"Completed {i + 1}/{len(self.test_tsne)} VLM predictions")
                
            except Exception as e:
                import traceback
                error_msg = f"VLM prediction failed for test point {i} (image: {test_image_paths[i] if i < len(test_image_paths) else 'N/A'}): {e}"
                logger.error(error_msg)
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Close figure if it exists
                if 'fig' in locals():
                    close_figure_safely(fig)
                # Use random prediction as fallback
                prediction = np.random.choice(self.unique_classes)
                predictions.append(prediction)
                
                # Store error details
                detailed_outputs.append({
                    'sample_index': i,
                    'test_point_coords': self.test_tsne[i].tolist() if i < len(self.test_tsne) else None,
                    'image_path': test_image_paths[i] if i < len(test_image_paths) else None,
                    'vlm_model': self.vlm_model_id,
                    'vlm_response': f"ERROR: {str(e)}",
                    'parsed_prediction': prediction,
                    'error': str(e),
                    'visualization_saved': False,
                    'backend_params': {
                        'use_pca_backend': self.use_pca_backend,
                        'use_3d_tsne': self.use_3d_tsne,
                        'use_knn_connections': self.use_knn_connections,
                        'knn_k': self.knn_k if self.use_knn_connections else None,
                        'zoom_factor': self.zoom_factor,
                        'max_train_plot_samples': self.max_train_plot_samples
                    }
                })
        
        return predictions, detailed_outputs
    
    
    def evaluate(
        self,
        test_image_paths: List[str],
        test_labels: List[int],
        return_detailed: bool = False,
        save_outputs: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the classifier on test data.
        
        Args:
            test_image_paths: List of test image paths
            test_labels: List of true test labels
            return_detailed: Whether to return detailed results
            save_outputs: Whether to save visualizations and responses
            output_dir: Directory to save outputs (required if save_outputs=True)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        start_time = time.time()
        
        # Make predictions with optional output saving
        predictions, detailed_outputs = self.predict(test_image_paths, save_outputs, output_dir)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import warnings
        
        accuracy = accuracy_score(test_labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'prediction_time': time.time() - start_time,
            'num_test_samples': len(test_labels)
        }
        
        if return_detailed:
            # Suppress sklearn warnings for undefined precision/recall
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")
                results.update({
                    'classification_report': classification_report(
                        test_labels, predictions, 
                        output_dict=True,
                        zero_division=0
                    ),
                    'confusion_matrix': confusion_matrix(test_labels, predictions),
                    'predictions': predictions,
                    'true_labels': test_labels,
                    'detailed_outputs': detailed_outputs[:20],  # Store first 20 for debugging
                    'visualizations_saved': save_outputs and output_dir,
                    'output_directory': output_dir if save_outputs else None
                })
                
        # Save detailed outputs as JSON if requested
        if save_outputs and output_dir and detailed_outputs:
            logger.info("Saving detailed prediction outputs...")
            outputs_file = os.path.join(output_dir, 'detailed_outputs.json')
            with open(outputs_file, 'w') as f:
                # Use the standardized JSON conversion utility
                json.dump(convert_for_json_serialization(detailed_outputs), f, indent=2)
        
        logger.info(f"CLAM image t-SNE classifier accuracy: {accuracy:.4f}")
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        return {
            'dinov2_model': self.dinov2_model,
            'embedding_size': self.embedding_size,
            'tsne_perplexity': self.tsne_perplexity,
            'tsne_n_iter': self.tsne_n_iter,
            'vlm_model_id': self.vlm_model_id,
            'use_3d_tsne': self.use_3d_tsne,
            'use_knn_connections': self.use_knn_connections,
            'knn_k': self.knn_k,
            'max_vlm_image_size': self.max_vlm_image_size,
            'image_dpi': self.image_dpi,
            'force_rgb_mode': self.force_rgb_mode,
            'zoom_factor': self.zoom_factor,
            'use_pca_backend': self.use_pca_backend,
            'max_train_plot_samples': self.max_train_plot_samples,
            'device': self.device
        }