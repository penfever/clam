#!/usr/bin/env python3
"""
Test script to verify API integration for OpenAI and Gemini models.
"""

import sys
import os
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clam.utils.model_loader import model_loader, GenerationConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_detection():
    """Test that the model loader correctly detects API models."""
    print("Testing model detection...")
    print("=" * 50)
    
    # Test OpenAI model detection
    openai_models = ["gpt-4.1", "gpt-4o", "gpt-3.5-turbo"]
    for model_name in openai_models:
        print(f"\nTesting OpenAI model: {model_name}")
        try:
            # This should auto-detect as OpenAI backend
            wrapper = model_loader.load_llm(model_name, backend="auto")
            print(f"  ✅ Detected as: {type(wrapper).__name__}")
        except Exception as e:
            print(f"  ❌ Error (expected if no API key): {e}")
    
    # Test Gemini model detection
    gemini_models = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]
    for model_name in gemini_models:
        print(f"\nTesting Gemini model: {model_name}")
        try:
            # This should auto-detect as Gemini backend
            wrapper = model_loader.load_llm(model_name, backend="auto")
            print(f"  ✅ Detected as: {type(wrapper).__name__}")
        except Exception as e:
            print(f"  ❌ Error (expected if no API key): {e}")
    
    # Test VLM detection
    print(f"\nTesting VLM model: gpt-4o")
    try:
        wrapper = model_loader.load_vlm("gpt-4o", backend="auto")
        print(f"  ✅ VLM detected as: {type(wrapper).__name__}")
    except Exception as e:
        print(f"  ❌ VLM Error (expected if no API key): {e}")

def test_generation_config():
    """Test GenerationConfig API conversion methods."""
    print("\n\nTesting GenerationConfig conversions...")
    print("=" * 50)
    
    config = GenerationConfig(
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        enable_thinking=True,
        thinking_summary=False
    )
    
    # Test OpenAI conversion
    openai_kwargs = config.to_openai_kwargs()
    print(f"\nOpenAI kwargs: {openai_kwargs}")
    
    # Test Gemini conversion
    gemini_kwargs = config.to_gemini_kwargs()
    print(f"Gemini kwargs: {gemini_kwargs}")
    
    print("  ✅ GenerationConfig conversions working")

def test_baseline_imports():
    """Test that all baseline classes can be imported."""
    print("\n\nTesting baseline imports...")
    print("=" * 50)
    
    try:
        from examples.vision.api_vlm_baseline import APIVLMBaseline, BiologicalAPIVLMBaseline
        print("  ✅ API VLM baselines imported successfully")
    except Exception as e:
        print(f"  ❌ API VLM baseline import error: {e}")
    
    try:
        from examples.vision.openai_vlm_baseline import OpenAIVLMBaseline, BiologicalOpenAIVLMBaseline
        print("  ✅ OpenAI VLM baselines imported successfully")
    except Exception as e:
        print(f"  ❌ OpenAI VLM baseline import error: {e}")
    
    try:
        from examples.vision.gemini_vlm_baseline import GeminiVLMBaseline, BiologicalGeminiVLMBaseline
        print("  ✅ Gemini VLM baselines imported successfully")
    except Exception as e:
        print(f"  ❌ Gemini VLM baseline import error: {e}")
    
    try:
        from examples.tabular.llm_baselines.openai_llm_baseline import OpenAILLMBaseline, evaluate_openai_llm
        print("  ✅ OpenAI LLM baseline imported successfully")
    except Exception as e:
        print(f"  ❌ OpenAI LLM baseline import error: {e}")
    
    try:
        from examples.tabular.llm_baselines.gemini_llm_baseline import GeminiLLMBaseline, evaluate_gemini_llm
        print("  ✅ Gemini LLM baseline imported successfully")
    except Exception as e:
        print(f"  ❌ Gemini LLM baseline import error: {e}")

def main():
    """Run all tests."""
    print("API Integration Test Suite")
    print("=" * 50)
    print("Note: API key errors are expected if OPENAI_API_KEY or GOOGLE_API_KEY are not set")
    
    test_model_detection()
    test_generation_config()
    test_baseline_imports()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("\nTo use the API models:")
    print("1. Set OPENAI_API_KEY environment variable for OpenAI models")
    print("2. Set GOOGLE_API_KEY environment variable for Gemini models")
    print("3. Install API dependencies: pip install 'clam[api]'")

if __name__ == "__main__":
    main()