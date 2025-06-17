# CLAM: CLassify Anything Model

CLAM (CLassify Anything Model) is a Python library for multi-modal classification using embeddings and Vision Language Models (VLMs). It supports images, audio, and tabular data through a unified interface.

## Features

### Multi-Modal Classification
- **Images**: DINOV2 embeddings → t-SNE/PCA → VLM classification
- **Audio**: Whisper/CLAP embeddings → t-SNE/PCA → VLM classification  
- **Tabular**: TabPFN embeddings → LLM fine-tuning

### Advanced Capabilities
- Vision Language Models (Qwen2.5-VL) for zero-shot classification
- Interactive t-SNE visualizations with KNN connections
- 3D visualizations and semantic class naming
- Vector Quantization (VQ) to reduce memory footprint
- Support for multiple datasets and evaluation frameworks

## Installation

### From PyPI (not yet available)

```bash
pip install clam
```

### From Source

First, clone the repository:

```bash
git clone https://github.com/penfever/clam.git
cd clam
```

Then install in development mode:

```bash
# Install in editable mode using the new pyproject.toml configuration
pip install -e .

# Install optional dependencies for different modalities
pip install -e ".[vision]"     # For image classification
pip install -e ".[audio]"      # For audio classification  
pip install -e ".[vlm]"        # For Vision Language Models
```

If you encounter any issues with imports, see troubleshooting instructions in `PACKAGING.md`.

## Quick Start

### Image Classification

```bash
# Test CIFAR-10 with CLAM t-SNE and baselines
python examples/vision/evaluate_all_vision.py --datasets cifar10 --models clam_tsne dinov2_linear qwen_vl --quick_test

# Test multiple datasets
python examples/vision/evaluate_all_vision.py --datasets cifar10 cifar100 --models clam_tsne --use_3d_tsne --use_knn_connections
```

### Audio Classification  

```bash
# Test ESC-50 and RAVDESS datasets
python examples/audio/evaluate_all_audio.py --datasets esc50 ravdess --models clam_tsne whisper_baseline

# Test with CLAP embeddings
python examples/audio/evaluate_all_audio.py --datasets esc50 --embedding_type clap --models clam_tsne
```

### Tabular Data

```python
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from clam.data import load_dataset, get_tabpfn_embeddings, create_llm_dataset
from clam.models import prepare_qwen_with_prefix_embedding
from clam.train import train_llm_with_tabpfn_embeddings, evaluate_llm_on_test_set
from clam.utils import setup_logging

# Set up logging
logger = setup_logging()

# 1. Load a dataset
dataset_name = 'har'  # Human Activity Recognition dataset
X, y, categorical_indicator, attribute_names, full_name = load_dataset(dataset_name)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 2. Get TabPFN embeddings
embedding_size = 1000
train_embeddings, val_embeddings, test_embeddings, tabpfn, y_train_sample = get_tabpfn_embeddings(
    X_train, y_train, X_val, X_test, embedding_size=embedding_size
)

# 3. Prepare Qwen model with prefix embedding
model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = prepare_qwen_with_prefix_embedding(embedding_size)

# 4. Create LLM dataset
output_dir = "./clam_output"
train_dataset, eval_dataset, test_dataset, label_encoder, prefix_data_file = create_llm_dataset(
    X_train, y_train_sample, X_val, y_val, X_test, y_test,
    train_embeddings, val_embeddings, test_embeddings,
    tokenizer, prefix_start_id, prefix_end_id, class_token_ids,
    output_dir=output_dir
)

# 5. Train LLM
trained_model, tokenizer = train_llm_with_tabpfn_embeddings(
    model, tokenizer, train_dataset, eval_dataset,
    prefix_start_id, prefix_end_id, class_token_ids, prefix_data_file, 
    output_dir=output_dir,
    num_train_epochs=3,
    max_train_samples=1000,  # Limit for faster training
)

# 6. Evaluate on test set
results = evaluate_llm_on_test_set(
    trained_model, tokenizer, test_dataset, 
    label_encoder, prefix_start_id, prefix_end_id,
    class_token_ids, prefix_data_file, max_test_samples=500
)

print(f"Test accuracy: {results['accuracy']:.4f}")
```

## Command-line Usage

CLAM provides a set of command-line scripts in the `examples` directory for easy training and evaluation:

### Training Models

```bash
# Basic usage with default 100 few-shot examples
python examples/tabular/train_tabular_dataset_tabular.py --dataset_name har --output_dir ./models/har_model

# Training with 50 few-shot examples instead of the default 100
python examples/tabular/train_tabular_dataset_tabular.py --dataset_name har --num_few_shot_examples 50

# Using example order permutation to discourage memorization
python examples/tabular/train_tabular_dataset_tabular.py --dataset_name har --permute_examples

# Using class-to-label mapping permutation to discourage memorization
python examples/tabular/train_tabular_dataset_tabular.py --dataset_name har --permute_labels

# Using variable few-shot examples to improve generalization
python examples/tabular/train_tabular_dataset_tabular.py --dataset_name har --variable_few_shot --few_shot_min 10 --few_shot_max 150

# Combining all generalization strategies
python examples/tabular/train_tabular_dataset_tabular.py --dataset_name har --permute_examples --permute_labels --variable_few_shot --few_shot_min 20 --few_shot_max 200
```

### Training with Vector Quantization

```bash
# Train a model using vector quantization
python examples/train_tabular_dataset_vq.py --dataset_name har --output_dir ./models/har_vq_model

# Train with custom VQ parameters
python examples/train_tabular_dataset_vq.py --dataset_name har --vq_num_embeddings 256 --vq_commitment_cost 0.2
```

### Evaluating Models

```bash
# Basic evaluation on a single dataset
python examples/evaluate_on_dataset.py --model_path ./models/har_model --dataset_name har

# Evaluating on multiple specific datasets
python examples/evaluate_on_dataset.py --model_path ./models/har_model --dataset_ids 1590,40975,37,54 --output_dir ./eval_results

# Evaluating on random datasets from OpenML
python examples/evaluate_on_dataset.py --model_path ./models/har_model --num_datasets 5 --output_dir ./eval_results
```

### Evaluating with Explanations

```bash
# Basic usage with a single dataset
python examples/evaluate_with_explanations.py --model_path ./models/har_model --dataset_name har

# Generating longer explanations
python examples/evaluate_with_explanations.py --model_path ./models/har_model --dataset_name har --max_explanation_tokens 100

# Adding counterfactual explanations
python examples/evaluate_with_explanations.py --model_path ./models/har_model --dataset_name har --explanation_type counterfactual

# Including feature importance analysis
python examples/evaluate_with_explanations.py --model_path ./models/har_model --dataset_name har --explanation_type feature_importance
```

### Evaluating Vector Quantized Models

```bash
# Basic usage with a single dataset
python examples/evaluate_with_vq.py --model_path ./models/vq_model --dataset_name har

# Compare with non-VQ model on a single dataset
python examples/evaluate_with_vq.py --model_path ./models/vq_model --compare_with ./models/standard_model --dataset_name har
```

## How It Works

CLAM Tabular leverages the power of TabPFN, a foundation model for tabular data, to extract meaningful embeddings from tabular datasets. These embeddings are then used as a prefix for a Large Language Model (LLM) to guide it toward understanding the tabular data patterns.

1. **Embedding Extraction**: TabPFN processes tabular data to generate embeddings that capture the data's statistical properties and patterns.

2. **Prefix Embedding Architecture**: The Qwen LLM is adapted with a custom embedding layer that allows it to process both embeddings from TabPFN and text tokens.

3. **Training Approach**: The LLM is fine-tuned to predict class labels based on the prefix embeddings, using techniques like mixup augmentation and class-balanced sampling.

4. **Inference Process**: At inference time, the model processes new data points by first converting them to embeddings with TabPFN, then using those embeddings as a prefix for the LLM to generate a class prediction.

### Vector Quantization Approach

CLAM Tabular also supports a Vector Quantization (VQ) approach that provides additional benefits:

1. **Discrete Representation**: VQ maps continuous embeddings to a discrete codebook, creating a more structured latent space.

2. **Reduced Memory Footprint**: By mapping to discrete codes, VQ reduces the memory requirements during training and inference.

3. **Improved Generalization**: Similar embeddings map to the same codebook vectors, which can improve model generalization.

4. **Alignment with LLM Architecture**: The discrete nature of VQ aligns better with the token-based architecture of language models.

## Advanced Usage

### Training with Vector Quantization

```python
from clam.data import load_dataset, get_tabpfn_embeddings, create_llm_dataset
from clam.models import prepare_qwen_with_vq_prefix_embedding
from clam.train import train_llm_with_tabpfn_embeddings

# Load dataset and get embeddings (same as standard approach)
X, y, categorical_indicator, attribute_names, full_name = load_dataset('har')
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
train_embeddings, val_embeddings, test_embeddings, tabpfn, y_train_sample = get_tabpfn_embeddings(
    X_train, y_train, X_val, X_test, embedding_size=1000
)

# Prepare model with vector quantization
model, tokenizer, prefix_start_id, prefix_end_id, class_token_ids = prepare_qwen_with_vq_prefix_embedding(
    embedding_size=1000,
    model_id="Qwen/Qwen2.5-1.5B-Instruct", 
    vq_num_embeddings=512,  # Size of the codebook
    vq_commitment_cost=0.25,  # Weight for the commitment loss
    vq_decay=0.99  # EMA decay factor for codebook updates
)

# Create dataset and train model (same as standard approach)
train_dataset, eval_dataset, test_dataset, label_encoder, prefix_data_file = create_llm_dataset(
    X_train, y_train_sample, X_val, y_val, X_test, y_test,
    train_embeddings, val_embeddings, test_embeddings,
    tokenizer, prefix_start_id, prefix_end_id, class_token_ids,
    output_dir="./vq_output"
)

trained_model, tokenizer = train_llm_with_tabpfn_embeddings(
    model, tokenizer, train_dataset, eval_dataset,
    prefix_start_id, prefix_end_id, class_token_ids, prefix_data_file, 
    output_dir="./vq_output"
)
```

### Important Note on Dataset Caching

CLAM includes a caching mechanism to improve performance by avoiding repeated loading of datasets that previously failed. If you need to force reloading of datasets that were previously skipped due to errors, you can use the `clear_failed_dataset_cache()` function:

```python
from clam.data.dataset import clear_failed_dataset_cache
clear_failed_dataset_cache()
```

This can be particularly useful if a dataset was temporarily unavailable or if there were connection issues during a previous run.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- TabPFN 0.1.7+
- NumPy
- Pandas
- scikit-learn
- datasets
- openml
- tqdm
- peft

## Citation

If you use CLAM in your research, please cite:

```
@software{feuer_clam_2025,
  author       = {Feuer, Benjamin and Liu, Yurong and Purucker, Lennart and Hegde, Chinmay},
  title        = {CLAM: Classify Anything Model},
  month        = june,
  year         = 2025
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
