#!/usr/bin/env python
"""
Test script to verify model saving and loading functionality for CLAM.
This test ensures that models can be properly saved and loaded 
with all necessary context for prefix embeddings.
"""

import os
import sys
import shutil
import tempfile
import unittest
import numpy as np
import torch

# Add parent directory to path to import clam
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clam.models import prepare_qwen_with_prefix_embedding
from clam.utils import setup_logging

class TestModelSaveLoad(unittest.TestCase):
    """Test case for model saving and loading functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.logger = setup_logging()
        self.embedding_size = 512  # Use a smaller size for faster testing
        
        # Create temporary directory for model saving/loading
        self.test_dir = tempfile.mkdtemp()
        self.logger.info(f"Using temporary directory: {self.test_dir}")
        
        # Create synthetic data for testing
        self.create_synthetic_data()
        
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
        self.logger.info(f"Removed temporary directory: {self.test_dir}")
        
    def create_synthetic_data(self):
        """Create synthetic embeddings data for testing."""
        # Create a small synthetic embedding tensor
        self.synthetic_embeddings = np.random.randn(10, self.embedding_size).astype(np.float32)
        self.synthetic_labels = np.random.randint(0, 5, size=10).astype(np.int64)
        
        # Save as npz file
        self.prefix_data_file = os.path.join(self.test_dir, "prefix_data.npz")
        np.savez(
            self.prefix_data_file,
            embeddings=self.synthetic_embeddings,
            class_labels=self.synthetic_labels
        )
        self.logger.info(f"Created synthetic data at {self.prefix_data_file}")
    
    def test_model_save_load(self):
        """Test that models can be saved and loaded with all necessary context."""
        self.logger.info("Testing model save and load")
        
        try:
            # 1. Prepare model
            self.logger.info("Step 1: Creating initial model")
            model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = prepare_qwen_with_prefix_embedding(
                embedding_size=self.embedding_size
            )
            
            # 2. Apply some modifications to check they're preserved
            self.logger.info("Step 2: Making modifications to the model")
            
            # Create a simple prompt
            prompt = "Predict the correct class for the given data.\n\n<PREFIX_START>_<PREFIX_END>\n\nLook at the data patterns and predict the class.\n\nThe class is:"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            
            # Make some dummy predictions to initialize weights differently
            with torch.no_grad():
                outputs = model(input_ids)
            
            # Save model weights
            self.logger.info("Step 3: Saving model")
            save_path = os.path.join(self.test_dir, "test_model")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            # Save some metadata about the model
            original_input_embeddings = model.base_model.get_input_embeddings().weight.data.clone()
            original_projector_weights = model.embedding_projector.weight.data.clone()
            
            # 3. Load model from saved weights
            self.logger.info("Step 4: Loading model from saved weights")
            loaded_model = model.__class__.from_pretrained(save_path)
            loaded_tokenizer = tokenizer.__class__.from_pretrained(save_path)
            
            # 4. Verify that all necessary parts were preserved
            self.logger.info("Step 5: Verifying model attributes were preserved")
            
            # Check model attributes
            self.assertEqual(loaded_model.prefix_start_id, prefix_start_id, 
                             "Prefix start ID was not preserved")
            self.assertEqual(loaded_model.prefix_end_id, prefix_end_id, 
                             "Prefix end ID was not preserved")
            
            # Compare class token IDs
            self.assertEqual(len(loaded_model.class_token_ids), len(class_token_ids), 
                             "Number of class token IDs doesn't match")
            for i, (orig, loaded) in enumerate(zip(class_token_ids, loaded_model.class_token_ids)):
                self.assertEqual(orig, loaded, f"Class token ID {i} doesn't match")
            
            # Check embedding size
            self.assertEqual(loaded_model.embedding_projector.in_features, self.embedding_size, 
                             "Embedding size was not preserved")
            
            # Check if tokenizer has the special tokens
            for token in ["<PREFIX_START>", "<PREFIX_END>", "<CLASS_0>"]:
                self.assertIn(token, loaded_tokenizer.get_vocab(),
                              f"Special token {token} not found in loaded tokenizer")
            
            # Verify weights were preserved (handle dtype and device differences)
            new_input_embeddings = loaded_model.base_model.get_input_embeddings().weight.data
            self.assertTrue(torch.allclose(original_input_embeddings.float().cpu(), new_input_embeddings.float().cpu()),
                           "Input embeddings were not preserved")
            
            new_projector_weights = loaded_model.embedding_projector.weight.data
            self.assertTrue(torch.allclose(original_projector_weights.float().cpu(), new_projector_weights.float().cpu()),
                           "Embedding projector weights were not preserved")
            
            # 5. Test inference with the loaded model
            self.logger.info("Step 6: Testing inference with loaded model")
            
            # Load prefix data
            prefix_data = np.load(self.prefix_data_file)
            prefix_embeddings = prefix_data['embeddings']
            prefix_class_labels = prefix_data['class_labels']
            
            # Convert to tensors
            device = next(loaded_model.parameters()).device
            prefix_embeddings_tensor = torch.tensor(prefix_embeddings, dtype=torch.float32).to(device)
            prefix_class_labels_tensor = torch.tensor(prefix_class_labels, dtype=torch.long).to(device)
            
            # Create a dummy query embedding
            query_embedding = torch.randn(self.embedding_size, dtype=torch.float32).to(device)
            
            # Test tokenizing
            placeholder_tokens = " ".join(["_"] * 10)
            sample_prompt = f"Predict the correct class for the given data.\n\n<PREFIX_START>{placeholder_tokens}<PREFIX_END>\n\nLook at the data patterns and predict the class.\n\nThe class is:"
            
            inputs = loaded_tokenizer(sample_prompt, return_tensors="pt").to(device)
            
            # Run inference
            with torch.no_grad():
                try:
                    # Forward pass through the model - we're not using the output, just checking it runs
                    outputs = loaded_model(inputs["input_ids"])
                    self.logger.info("Successfully ran inference with loaded model")
                    
                    # If we want to measure something more concrete, we could check logits shape
                    self.assertEqual(outputs.logits.shape[0], 1, "Batch size doesn't match")
                    self.assertEqual(outputs.logits.shape[1], inputs["input_ids"].shape[1], "Sequence length doesn't match")
                except Exception as e:
                    self.fail(f"Failed to run inference with loaded model: {e}")
                    
            self.logger.info("All model save and load tests passed!")
            
        except Exception as e:
            self.logger.error(f"Error in test_model_save_load: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise


class TestFullPipeline(unittest.TestCase):
    """Test case for full model training, saving, loading, and evaluation pipeline."""
    
    @unittest.skip("This test requires significant computational resources and is meant for manual runs")
    def test_full_pipeline(self):
        """
        Test full pipeline of:
        1. Training a model with prefix embeddings
        2. Saving the model
        3. Loading the model
        4. Evaluating the model
        
        This test is skipped by default as it requires significant computational resources.
        """
        from clam.data import load_dataset, get_tabpfn_embeddings, create_llm_dataset
        from clam.train import train_llm_with_tabpfn_embeddings, evaluate_llm_on_test_set
        
        logger = setup_logging()
        logger.info("Testing full pipeline")
        
        # Create temp directory
        test_dir = tempfile.mkdtemp()
        logger.info(f"Using temporary directory: {test_dir}")
        
        try:
            # 1. Setup parameters
            embedding_size = 512
            dataset_name = 'har'
            output_dir = os.path.join(test_dir, "model_output")
            os.makedirs(output_dir, exist_ok=True)
            
            # 2. Load dataset (using the smallest available)
            logger.info("Loading dataset")
            X, y, categorical_indicator, attribute_names, full_name = load_dataset(dataset_name)
            
            # Split into train, validation, and test (using very small samples)
            from sklearn.model_selection import train_test_split
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
            
            # Take only a very small sample for quick testing
            X_train = X_train[:100]
            y_train = y_train[:100]
            X_val = X_val[:20]
            y_val = y_val[:20]
            X_test = X_test[:20]
            y_test = y_test[:20]
            
            # 3. Get TabPFN embeddings
            train_embeddings, val_embeddings, test_embeddings, tabpfn, y_train_sample = get_tabpfn_embeddings(
                X_train, y_train, X_val, X_test, 
                embedding_size=embedding_size,
                max_samples=100
            )
            
            # 4. Prepare model
            model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = prepare_qwen_with_prefix_embedding(
                embedding_size=embedding_size
            )
            
            # 5. Create LLM dataset
            train_dataset, eval_dataset, test_dataset, label_encoder, prefix_data_file = create_llm_dataset(
                X_train, y_train_sample, X_val, y_val, X_test, y_test,
                train_embeddings, val_embeddings, test_embeddings,
                tokenizer, prefix_start_id, prefix_end_id, class_token_ids,
                output_dir=output_dir
            )
            
            # 6. Train with minimal settings
            trained_model, tokenizer = train_llm_with_tabpfn_embeddings(
                model, tokenizer, train_dataset, eval_dataset,
                prefix_start_id, prefix_end_id, class_token_ids, prefix_data_file, 
                output_dir=output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                max_train_samples=10,
                save_best_model=True,
            )
            
            # 7. Get the model path (could be best_model or final_model)
            model_path = os.path.join(output_dir, "best_model")
            if not os.path.exists(model_path):
                model_path = os.path.join(output_dir, "final_model")
            
            # 8. Load the model from disk
            loaded_model = model.__class__.from_pretrained(model_path)
            loaded_tokenizer = tokenizer.__class__.from_pretrained(model_path)
            
            # 9. Evaluate the loaded model
            results = evaluate_llm_on_test_set(
                loaded_model, loaded_tokenizer, test_dataset, 
                label_encoder, prefix_start_id, prefix_end_id,
                class_token_ids, prefix_data_file, max_test_samples=10
            )
            
            # 10. Check results
            self.assertIn('accuracy', results, "Results should include accuracy")
            
            logger.info(f"Full pipeline test completed successfully with accuracy: {results['accuracy']}")
            
        finally:
            # Clean up
            shutil.rmtree(test_dir)
            logger.info(f"Cleaned up temporary directory: {test_dir}")


if __name__ == "__main__":
    # Run only the basic model save/load test by default
    suite = unittest.TestSuite()
    suite.addTest(TestModelSaveLoad('test_model_save_load'))
    
    # To run the full pipeline test, uncomment below:
    # suite.addTest(TestFullPipeline('test_full_pipeline'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)