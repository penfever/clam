#!/usr/bin/env python
"""
Quick test script to verify model saving and loading functionality for CLAM.
This is a simplified version of the full test_model_save_load.py, designed
to run quickly with minimal dependencies and resources.
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import torch

# Add parent directory to path to import clam
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clam.models import prepare_qwen_with_prefix_embedding
from clam.utils import setup_logging

def run_quick_test():
    """Run a quick test of model saving and loading."""
    logger = setup_logging()
    logger.info("Starting quick test of model save/load functionality")
    
    # Create temporary directory
    test_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {test_dir}")
    
    try:
        # Use a small embedding size for faster testing
        embedding_size = 256
        
        # 1. Prepare model
        logger.info("Step 1: Creating model")
        model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = prepare_qwen_with_prefix_embedding(
            embedding_size=embedding_size
        )
        
        # Verify model has the embedding projector
        if not hasattr(model, 'embedding_projector'):
            logger.error("Model does not have an embedding_projector attribute!")
            return False
        
        # Create a dummy embedding
        dummy_embedding = torch.zeros(embedding_size, dtype=torch.float32)
        with torch.no_grad():
            # Test the embedding projector
            projected = model.embedding_projector(dummy_embedding)
            logger.info(f"Embedding projector output shape: {projected.shape}")
        
        # 2. Save model
        logger.info("Step 2: Saving model")
        save_path = os.path.join(test_dir, "test_model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        # Verify model was saved
        model_files = os.listdir(save_path)
        if "embedding_projector.pt" not in model_files or "model_info.pt" not in model_files:
            logger.error(f"Missing critical files in saved model. Found: {model_files}")
            return False
        
        logger.info(f"Model saved successfully. Files: {', '.join(model_files)}")
        
        # 3. Load model
        logger.info("Step 3: Loading model")
        loaded_model = model.__class__.from_pretrained(save_path)
        loaded_tokenizer = tokenizer.__class__.from_pretrained(save_path)
        
        # 4. Verify attributes
        logger.info("Step 4: Verifying model attributes")
        tests_passed = True
        
        # Test basic attributes
        if loaded_model.prefix_start_id != prefix_start_id:
            logger.error(f"Prefix start ID mismatch: {loaded_model.prefix_start_id} != {prefix_start_id}")
            tests_passed = False
            
        if loaded_model.prefix_end_id != prefix_end_id:
            logger.error(f"Prefix end ID mismatch: {loaded_model.prefix_end_id} != {prefix_end_id}")
            tests_passed = False
            
        if loaded_model.embedding_projector.in_features != embedding_size:
            logger.error(f"Embedding size mismatch: {loaded_model.embedding_projector.in_features} != {embedding_size}")
            tests_passed = False
            
        if len(loaded_model.class_token_ids) != len(class_token_ids):
            logger.error(f"Class token IDs count mismatch: {len(loaded_model.class_token_ids)} != {len(class_token_ids)}")
            tests_passed = False
        else:
            for i, (orig, loaded) in enumerate(zip(class_token_ids, loaded_model.class_token_ids)):
                if orig != loaded:
                    logger.error(f"Class token ID {i} mismatch: {loaded} != {orig}")
                    tests_passed = False
                    
        # Test tokenizer special tokens
        special_tokens = ["<PREFIX_START>", "<PREFIX_END>"] + [f"<CLASS_{i}>" for i in range(10)]
        for token in special_tokens:
            if token not in loaded_tokenizer.get_vocab():
                logger.error(f"Special token {token} not found in loaded tokenizer")
                tests_passed = False
                
        # 5. Test running the model
        logger.info("Step 5: Testing model inference")
        try:
            # Create a simple prompt
            prompt = "Predict the correct class for the given data.\n\n<PREFIX_START>_<PREFIX_END>\n\nLook at the data patterns and predict the class.\n\nThe class is:"
            
            # Tokenize
            inputs = loaded_tokenizer(prompt, return_tensors="pt")
            
            # Run the model
            with torch.no_grad():
                outputs = loaded_model(inputs["input_ids"])
                
            logger.info(f"Model inference successful, output shape: {outputs.logits.shape}")
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            tests_passed = False
        
        # Print final result
        if tests_passed:
            logger.info("✅ All tests passed! Model save/load functionality is working correctly.")
            return True
        else:
            logger.error("❌ Some tests failed. See above for details.")
            return False
            
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
        
    finally:
        # Clean up
        shutil.rmtree(test_dir)
        logger.info(f"Cleaned up temporary directory: {test_dir}")

if __name__ == "__main__":
    success = run_quick_test()
    # Exit with appropriate code
    sys.exit(0 if success else 1)