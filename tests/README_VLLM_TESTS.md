# VLLM Backend Tests

This directory contains comprehensive tests for the VLLM backend functionality in the CLAM codebase, specifically designed for Mac M4 with 24GB of VRAM.

## Test Files

### 1. `test_vllm_backend.py`
Comprehensive test suite covering all aspects of VLLM backend functionality:

- **VLLM Installation**: Verifies VLLM can be imported and initialized
- **Model Loader Integration**: Tests the centralized model loader with VLLM backend
- **Basic Text Generation**: Validates that VLLM can generate text responses
- **VRAM Usage**: Monitors memory usage on Mac M4 (MPS backend)
- **Backend Selection**: Tests automatic backend selection (VLLM vs transformers)
- **TabLLM Integration**: Validates tabular LLM baselines work with VLLM
- **Examples Compatibility**: Ensures examples functionality is preserved

### 2. `test_vllm_tabular_integration.py`
Focused integration tests for tabular LLM functionality:

- **Tabular Model Loading**: Tests loading LLMs specifically for tabular tasks
- **Data Serialization**: Validates conversion of tabular data to text prompts
- **Classification Prompts**: Tests generation of classification responses
- **Batch Processing**: Validates batch inference capabilities
- **Memory Monitoring**: Specific tests for Mac M4 VRAM usage
- **Examples Integration**: Compatibility with existing tabular examples

## Hardware Requirements

- **Platform**: Mac M4 (tested on Darwin arm64)
- **Memory**: 24GB VRAM available
- **Python**: 3.10+ (tested with 3.10.16)
- **Environment**: conda environment named "llata"

## Dependencies

The tests automatically check for and validate:

- `vllm >= 0.9.1`
- `torch >= 2.7.0` with MPS support
- `transformers >= 4.51.1`
- Standard ML libraries (numpy, pandas, scikit-learn)

## Running the Tests

### Prerequisites
```bash
# Activate the conda environment
conda activate llata

# Ensure VLLM is installed
pip install vllm
```

### Basic VLLM Backend Tests
```bash
cd /path/to/clam
python tests/test_vllm_backend.py
```

### Tabular Integration Tests
```bash
cd /path/to/clam
python tests/test_vllm_tabular_integration.py
```

### Quick Validation
```bash
cd /path/to/clam
python -c "
import sys; sys.path.insert(0, '.')
from clam.utils.model_loader import model_loader, GenerationConfig
model_wrapper = model_loader.load_llm('microsoft/DialoGPT-small', backend='vllm', max_model_len=64)
result = model_wrapper.generate('Hello', GenerationConfig(max_new_tokens=5))
print(f'VLLM works: {result}')
model_wrapper.unload()
"
```

## Expected Results

### Successful Test Output
All tests should pass with output similar to:
```
=== VLLM Backend Validation Summary ===
✓ VLLM 0.9.1 installed and importable
✓ Model loader VLLM integration works
✓ Basic VLLM text generation works
✓ Examples functionality compatible
✓ Running on Darwin arm64
✓ MPS (Metal) backend available for Mac M4 GPU acceleration

🎉 VLLM backend validation completed successfully!
```

### Performance Characteristics

On Mac M4 with 24GB VRAM, you should observe:

- **Model Loading**: ~5-10 seconds for small models (DialoGPT-small)
- **Text Generation**: ~25-50 tokens/second for inference
- **Memory Usage**: Models load into system memory (not discrete GPU VRAM)
- **Backend**: Automatic selection of VLLM for supported models
- **Fallback**: Graceful fallback to transformers when needed

## Key Features Validated

### 1. VLLM Backend Integration
- ✅ VLLM loads and initializes correctly
- ✅ Model wrapper abstraction works seamlessly
- ✅ Generation configurations convert properly
- ✅ Batch inference supported

### 2. Tabular LLM Support
- ✅ Tabular data can be serialized to text prompts
- ✅ Classification tasks work with VLLM backend
- ✅ Few-shot learning examples supported
- ✅ Existing examples remain functional

### 3. Mac M4 Optimization
- ✅ MPS (Metal Performance Shaders) detected and available
- ✅ Memory usage monitored (system RAM, not discrete VRAM)
- ✅ CPU fallback works when GPU unavailable
- ✅ Optimal performance for local inference

### 4. Core Functionality Preservation
- ✅ Examples in `examples/tabular/` work unchanged
- ✅ Model loader automatically selects best backend
- ✅ Configuration and generation parameters preserved
- ✅ Error handling and fallbacks functional

## Troubleshooting

### Common Issues

1. **VLLM Import Error**
   ```bash
   pip install vllm
   # If build fails, try:
   pip install vllm --no-build-isolation
   ```

2. **Memory Issues**
   - Reduce `max_model_len` parameter
   - Lower `gpu_memory_utilization` setting
   - Use smaller models for testing

3. **MPS Not Available**
   - Ensure macOS 12.3+ and compatible hardware
   - Falls back to CPU automatically
   - Performance will be slower but functional

4. **Model Loading Timeout**
   - First run downloads models from HuggingFace
   - Subsequent runs use cached models
   - Large models may take longer to load

### Debug Mode
Run tests with verbose logging:
```bash
PYTHONPATH=. python -v tests/test_vllm_backend.py
```

## Contributing

When adding new VLLM-related functionality:

1. Add tests to validate the new feature
2. Ensure Mac M4 compatibility
3. Test both VLLM and transformers backends
4. Validate examples functionality still works
5. Update this README with new test descriptions

## References

- [VLLM Documentation](https://docs.vllm.ai/)
- [Mac Metal Performance Shaders](https://developer.apple.com/metal/pytorch/)
- [CLAM Tabular Examples](../examples/tabular/)
- [Model Loader Documentation](../clam/utils/model_loader.py)