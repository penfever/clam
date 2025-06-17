# CLAM Tests

This directory contains test scripts for the CLAM library.

## Test Status

### ✅ Working Tests

#### Core Functionality Tests
- **`test_install.py`**: Tests the installation of the CLAM package and verifies all imports work correctly.
  ```bash
  python tests/test_install.py
  ```

- **`test_clam_tsne_extraction.py`**: Tests CLAM t-SNE metric extraction, variable extraction, and W&B data handling.
  ```bash
  python tests/test_clam_tsne_extraction.py
  ```

- **`test_knn_fix.py`**: Tests the fixed KNN functionality for t-SNE visualizations.
  ```bash
  python tests/test_knn_fix.py
  ```

- **`test_embedding_cache.py`**: Tests embedding cache loading functionality with limits and multi-ensemble support.
  ```bash
  python tests/test_embedding_cache.py
  ```

#### Audio Tests
- **`test_baselines_simple.py`**: Tests audio baseline classifiers (Whisper KNN and CLAP zero-shot). Requires `msclap` library.
  ```bash
  pip install msclap
  python tests/test_baselines_simple.py
  ```

- **`test_minimal_audio.py`**: Tests minimal CLAM audio functionality with synthetic data.
  ```bash
  python tests/test_minimal_audio.py
  ```

### ⚠️ Partially Working Tests

- **`test_audio_simple.py`**: Tests audio processing pipeline. Some parts work but has import issues with examples module.
  ```bash
  python tests/test_audio_simple.py
  ```

### ❌ Failing Tests

- **`test_model_save_load.py`**: Tests model save/load functionality. Currently fails due to BFloat16/Float dtype mismatch.
- **`test_embedding_determinism.py`**: Tests embedding generation determinism. Fails because it references old file paths.

### 🗑️ Deprecated/Analysis Scripts

These are analysis scripts rather than tests:
- `analyze_embeddings.py` - Embedding analysis utility
- `analyze_embeddings_all.py` - Bulk embedding analysis
- `analyze_json_schemas.py` - JSON schema analysis
- `check_git_files.py` - Git file checking utility
- `setup_fix.py` - Setup fixing utility
- `transform_cc18_to_tabarena_complete.py` - Data transformation script

## Running Tests

### Run All Working Tests
```bash
# Core tests
python tests/test_install.py
python tests/test_clam_tsne_extraction.py
python tests/test_knn_fix.py
python tests/test_embedding_cache.py

# Audio tests (requires msclap)
pip install msclap
python tests/test_baselines_simple.py
python tests/test_minimal_audio.py
```

### Run with pytest
```bash
# Install pytest if not available
pip install pytest

# Run specific working tests
python -m pytest tests/test_install.py -v
python -m pytest tests/test_clam_tsne_extraction.py -v
python -m pytest tests/test_knn_fix.py -v
python -m pytest tests/test_embedding_cache.py -v
```

## Dependencies

### Required for Audio Tests
```bash
pip install msclap  # For CLAP zero-shot classifier
```

### Required for All Tests
All core dependencies are installed with the main CLAM package:
```bash
pip install -e .
```

## Test Coverage

The working tests cover:

1. **Package Installation**: Verify CLAM package installs correctly and all imports work
2. **Core Data Processing**: t-SNE metric extraction and variable handling
3. **Visualization**: KNN connections and t-SNE plotting
4. **Caching**: Embedding cache loading with various configurations
5. **Audio Pipeline**: Whisper embeddings, audio baselines, and CLAM audio classification
6. **Utility Functions**: Various helper functions and data processing

## Writing New Tests

When adding new features to CLAM, please add corresponding tests to this directory. Test files should:

1. Follow the naming convention `test_*.py`
2. Include docstrings explaining what is being tested
3. Use descriptive test function names
4. Handle cleanup of any temporary files/directories
5. Include proper error handling and informative error messages

## Troubleshooting

### Import Errors
If you get import errors for the `examples` module, make sure you're running tests from the project root directory and that the CLAM package is installed in editable mode:
```bash
cd /path/to/clam
pip install -e .
python tests/test_name.py
```

### Missing Dependencies
Some tests require additional dependencies:
- Audio tests require `msclap`: `pip install msclap`
- Some numpy version conflicts may occur with `msclap` - this is expected

### Timeout Issues
Some tests may take time due to model loading. The audio tests download models on first run.