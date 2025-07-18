"""
Qwen Vision-Language Model baseline for image classification.
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from sklearn.metrics import accuracy_score
import time
import os
import sys
import json
from PIL import Image
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from clam.utils.vlm_prompting import create_direct_classification_prompt, parse_vlm_response, create_vlm_conversation

logger = logging.getLogger(__name__)


class QwenVLBaseline:
    """
    Qwen2.5-VL baseline for image classification.
    Uses a Vision-Language Model for zero-shot or few-shot image classification.
    
    Can be used with either DataLoaders or image paths directly.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None,
                 model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: Optional[str] = None,
                 use_semantic_names: bool = False):
        self.num_classes = num_classes
        self.class_names = class_names or []
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_semantic_names = use_semantic_names
        self.is_fitted = False
        self.raw_responses = []  # Store raw VLM responses for analysis
        
    def load_model(self):
        """Load Qwen VL model."""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            
            logger.info(f"Loading Qwen VL model: {self.model_name}")
            
            # Mac-compatible loading
            if sys.platform == "darwin":
                logger.info("Mac detected: using CPU and float32 for VLM")
                torch_dtype = torch.float32
                device_map = "cpu"
            else:
                torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                device_map = "auto" if torch.cuda.is_available() else "cpu"
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            logger.info("Qwen VL model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Qwen VL model requires transformers library: {e}")
            logger.warning("Install with: pip install -e '.[vlm]'")
            # Fallback to mock implementation for testing
            logger.warning("Using mock Qwen VL implementation for testing")
            self.model = "mock"
            self.processor = "mock"
        except Exception as e:
            logger.error(f"Failed to load Qwen VL model: {e}")
            # Fallback to mock implementation for testing
            logger.warning("Using mock Qwen VL implementation for testing")
            self.model = "mock"
            self.processor = "mock"
    
    def fit(self, train_data=None, train_labels=None, class_names: Optional[List[str]] = None) -> 'QwenVLBaseline':
        """
        Fit the VLM (no training needed for zero-shot).
        
        Args:
            train_data: Either DataLoader or List[str] of image paths (unused, for API compatibility)
            train_labels: Training labels (unused, for API compatibility)
            class_names: Optional list of class names
        """
        if self.model is None:
            self.load_model()
        if class_names:
            self.class_names = class_names
        self.is_fitted = True
        return self
    
    def predict(self, test_data) -> np.ndarray:
        """
        Predict using Qwen VL model.
        
        Args:
            test_data: Either DataLoader or List[str] of image paths
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(test_data, DataLoader):
            return self._predict_from_loader(test_data)
        else:
            return self._predict_from_paths(test_data)
    
    def _predict_from_loader(self, test_loader: DataLoader) -> np.ndarray:
        """Predict using Qwen VL model from DataLoader."""
        predictions = []
        
        for batch_images, _ in test_loader:
            for i in range(len(batch_images)):
                # Convert tensor to PIL Image
                image_tensor = batch_images[i]
                # Denormalize if needed (ImageNet normalization)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image_tensor = image_tensor * std + mean
                image_tensor = torch.clamp(image_tensor, 0, 1)
                
                # Convert to PIL
                import torchvision.transforms as transforms
                image = transforms.ToPILImage()(image_tensor)
                
                # Predict
                if self.model == "mock":
                    prediction = np.random.randint(0, self.num_classes)
                else:
                    prediction = self._predict_single_image(image, image_path=f"batch_{i}_sample_{i}", use_semantic_names=self.use_semantic_names)
                predictions.append(prediction)
        
        return np.array(predictions)
    
    def _predict_from_paths(self, test_paths: List[str]) -> np.ndarray:
        """Predict using Qwen VL model from image paths."""
        predictions = []
        
        logger.info(f"Running Qwen VL predictions on {len(test_paths)} images")
        
        for i, image_path in enumerate(test_paths):
            if i % 100 == 0:
                logger.info(f"Processing image {i+1}/{len(test_paths)}")
            
            try:
                if self.model == "mock":
                    # Mock prediction for testing
                    prediction = np.random.randint(0, self.num_classes)
                else:
                    image = Image.open(image_path).convert('RGB')
                    prediction = self._predict_single_image(image, image_path=image_path, use_semantic_names=self.use_semantic_names)
                
                predictions.append(prediction)
                
            except Exception as e:
                import traceback
                error_msg = f"Error processing image {image_path}: {e}"
                logger.error(error_msg)
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Default to random prediction
                predictions.append(np.random.randint(0, self.num_classes))
        
        return np.array(predictions)
    
    def _predict_single_image(self, image: Image.Image, modality: str = "image", dataset_description: Optional[str] = None, 
                              image_path: Optional[str] = None, use_semantic_names: bool = False) -> int:
        """Predict single image using VLM with unified prompting strategy."""
        if not self.class_names:
            raise ValueError("Class names must be provided for VLM prediction")
        
        # Create direct image classification prompt using centralized function
        prompt_text = create_direct_classification_prompt(
            class_names=self.class_names,
            dataset_description=dataset_description,
            use_semantic_names=use_semantic_names
        )
        
        # Create conversation using vlm_prompting utilities
        conversation = create_vlm_conversation(image, prompt_text)
        
        # Process conversation format
        text_prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt", padding=True)
        
        # Move to device if needed
        if self.device != "cpu" and hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate response with proper sampling configuration
        # Note: temperature is only used when do_sample=True
        generation_config = {
            "max_new_tokens": 1024,
            "do_sample": True,  # Enable sampling to use temperature
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        # Extract only the generated part (exclude the input prompt)
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        # Store raw response for analysis
        response_entry = {
            "image_path": image_path,
            "prompt": prompt_text,
            "raw_response": response,
            "timestamp": time.time()
        }
        self.raw_responses.append(response_entry)
        
        # Log first 10 responses for debugging
        if len(self.raw_responses) <= 10:
            logger.info(f"Qwen VL Response #{len(self.raw_responses)} for {image_path}:")
            logger.info(f"  Prompt: {prompt_text[:200]}...")
            logger.info(f"  Raw Response: '{response}'")
            logger.info(f"  Model: {self.model_name}")
            logger.info("  " + "-"*50)
        
        # Parse response using vlm_prompting utilities
        predicted_class = parse_vlm_response(
            response, 
            unique_classes=self.class_names,
            logger_instance=logger,
            use_semantic_names=use_semantic_names
        )
        
        # Add parsed result to response entry
        response_entry["parsed_class"] = predicted_class
        
        # Convert to class index
        for i, class_name in enumerate(self.class_names):
            if predicted_class == class_name:
                response_entry["predicted_index"] = i
                return i
        
        # Default to first class if no match
        response_entry["predicted_index"] = 0
        return 0
    
    def save_raw_responses(self, output_path: str, benchmark_name: str = "unknown"):
        """Save raw VLM responses to JSON file."""
        if not self.raw_responses:
            logger.warning("No raw responses to save")
            return
        
        # Create output data with metadata
        output_data = {
            "metadata": {
                "benchmark": benchmark_name,
                "model_name": self.model_name,
                "num_responses": len(self.raw_responses),
                "class_names": self.class_names,
                "num_classes": self.num_classes,
                "saved_at": time.time()
            },
            "responses": self.raw_responses
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.raw_responses)} raw responses to {output_path}")
    
    def evaluate(self, test_data, test_labels: List[int], save_raw_responses: bool = False, 
                 output_dir: Optional[str] = None, benchmark_name: str = "unknown") -> Dict[str, Any]:
        """
        Evaluate VLM on test data.
        
        Args:
            test_data: Either DataLoader or List[str] of image paths
            test_labels: Ground truth labels
            save_raw_responses: Whether to save raw VLM responses
            output_dir: Directory to save raw responses (if save_raw_responses=True)
            benchmark_name: Name of the benchmark for the raw responses file
            
        Returns:
            Dictionary with evaluation metrics
        """
        start_time = time.time()
        predictions = self.predict(test_data)
        
        accuracy = accuracy_score(test_labels, predictions)
        
        # Save raw responses if requested
        if save_raw_responses and output_dir:
            raw_responses_filename = f"{benchmark_name}_qwenvl_raw_responses.json"
            raw_responses_path = os.path.join(output_dir, raw_responses_filename)
            self.save_raw_responses(raw_responses_path, benchmark_name)
        
        return {
            'accuracy': accuracy,
            'prediction_time': time.time() - start_time,
            'num_test_samples': len(test_labels),
            'predictions': predictions,
            'true_labels': test_labels,
            'num_raw_responses': len(self.raw_responses) if save_raw_responses else 0
        }


class BiologicalQwenVLBaseline(QwenVLBaseline):
    """
    Specialized Qwen VL baseline for biological image classification.
    Uses domain-specific prompts for biological organisms.
    """
    
    def _predict_single_image(self, image: Image.Image, dataset_description: Optional[str] = None,
                              image_path: Optional[str] = None) -> int:
        """Predict single image using VLM with biological-specific prompt."""
        # Create biological-specific dataset description if not provided
        if dataset_description is None:
            dataset_description = "Biological image classification dataset with various organisms and specimens"
        
        # Call parent method with biological context
        return super()._predict_single_image(
            image, 
            modality="image",
            dataset_description=dataset_description,
            image_path=image_path,
            use_semantic_names=getattr(self, 'use_semantic_names', False)
        )