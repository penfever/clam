# LLATA Tests

This directory contains test scripts for the LLATA library.

## Test Files

### Model Save/Load Tests

- **quick_test_save_load.py**: A quick test of model saving and loading functionality. This script creates a model, saves it, loads it, and verifies that all necessary components are preserved.

  ```bash
  python quick_test_save_load.py
  ```

- **test_model_save_load.py**: A more comprehensive test suite for model saving and loading. This includes unit tests that verify all model attributes and functionality are preserved during saving and loading.

  ```bash
  python test_model_save_load.py
  ```

### Other Tests

- **test_install.py**: Tests the installation of the LLATA package.

## Running Tests

You can run individual test scripts directly:

```bash
python tests/quick_test_save_load.py
```

Or run all tests with unittest:

```bash
python -m unittest discover tests
```

## Test Coverage

The tests cover:

1. **Model Creation**: Verify that models can be created with the correct attributes and functionality.
2. **Model Saving**: Test that models are saved with all necessary components, including:
   - Base model weights
   - Embedding projector weights
   - Special token IDs (prefix_start_id, prefix_end_id, class_token_ids)
   - Tokenizer configuration

3. **Model Loading**: Test that models can be loaded from saved weights with all components preserved.
4. **Model Inference**: Verify that loaded models can perform inference correctly.

## Writing New Tests

When adding new features to LLATA, please add corresponding tests to this directory. Test files should follow the naming convention `test_*.py` to be automatically discovered by unittest.