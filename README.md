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

## API Models Support

CLAM now supports the latest OpenAI and Gemini models through their respective APIs, providing access to cutting-edge AI capabilities without requiring local GPU resources.

## Supported Models (2025)

### OpenAI Models

#### LLM Models (for tabular data)
- **GPT-4.1**: Latest flagship model with 1M token context
- **GPT-4.1 mini**: 83% cheaper, nearly half the latency
- **GPT-4.1 nano**: Fastest and cheapest, 1M token context
- **GPT-4o**: Previous flagship model, still excellent
- **GPT-3.5 Turbo**: Cost-effective for simpler tasks
- **o3, o4-mini**: Latest reasoning models

#### VLM Models (for vision tasks)
- **GPT-4.1**: Vision capabilities with latest improvements
- **GPT-4o**: Excellent multimodal performance

### Gemini Models

#### LLM/VLM Models (support both text and vision)
- **Gemini 2.5 Pro**: Highest intelligence, thinking capabilities
- **Gemini 2.5 Flash**: Workhorse model, fast performance
- **Gemini 2.5 Flash-Lite**: Lowest latency and cost
- **Gemini 2.0 Flash**: Previous generation, still capable
- **Gemini 2.0 Pro Experimental**: Experimental features

All Gemini 2.x models support:
- Thinking capabilities for enhanced reasoning
- Thought summaries for transparency
- Native multimodal processing (text + images)

## Installation

Install the API dependencies:

```bash
pip install 'clam[api]'
```

This installs:
- `openai>=1.15.0` - OpenAI API client
- `google-generativeai>=0.8.0` - Gemini API client
- `httpx>=0.26.0` - HTTP client for reliability
- `tenacity>=8.2.0` - Retry logic for API calls
- `pydantic>=2.5.0` - Structured API responses

## Setup

### OpenAI Setup
1. Get an API key from [OpenAI Platform](https://platform.openai.com/)
2. Set the environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### Gemini Setup
1. Get an API key from [Google AI Studio](https://aistudio.google.com/)
2. Set the environment variable:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

## Usage

### Automatic Model Detection

CLAM automatically detects API models and selects the appropriate backend:

```python
from clam.utils.model_loader import model_loader

# Automatically detects as OpenAI
llm = model_loader.load_llm("gpt-4o")

# Automatically detects as Gemini
llm = model_loader.load_llm("gemini-2.5-flash")

# Automatically detects as OpenAI VLM
vlm = model_loader.load_vlm("gpt-4.1")

# Automatically detects as Gemini VLM
vlm = model_loader.load_vlm("gemini-2.5-pro")
```

### Vision Tasks

#### Unified API VLM Baseline
```python
from examples.vision.api_vlm_baseline import APIVLMBaseline

# Works with any supported API model
model = APIVLMBaseline(
    num_classes=10,
    class_names=["cat", "dog", "bird", ...],
    model_name="gpt-4o",  # or "gemini-2.5-flash"
    use_semantic_names=True
)

model.fit(train_data, train_labels)
predictions = model.predict(test_images)
```

#### OpenAI-specific VLM
```python
from examples.vision.openai_vlm_baseline import OpenAIVLMBaseline

model = OpenAIVLMBaseline(
    num_classes=1000,
    model_name="gpt-4.1",  # Latest OpenAI vision model
    use_semantic_names=True
)
```

#### Gemini-specific VLM with Thinking
```python
from examples.vision.gemini_vlm_baseline import GeminiVLMBaseline

model = GeminiVLMBaseline(
    num_classes=1000,
    model_name="gemini-2.5-pro",
    enable_thinking=True  # Use thinking capabilities
)
```

### Tabular Tasks

#### OpenAI LLM for Tabular Data
```python
from examples.tabular.llm_baselines.openai_llm_baseline import OpenAILLMBaseline

model = OpenAILLMBaseline(
    num_classes=3,
    class_names=["low", "medium", "high"],
    model_name="gpt-4.1-mini",  # Cost-effective choice
    feature_names=["age", "income", "score"]
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### Gemini LLM for Tabular Data
```python
from examples.tabular.llm_baselines.gemini_llm_baseline import GeminiLLMBaseline

model = GeminiLLMBaseline(
    num_classes=2,
    model_name="gemini-2.5-flash",
    enable_thinking=True  # Enhanced reasoning
)
```

### Advanced Configuration

#### Custom Generation Settings
```python
from clam.utils.model_loader import GenerationConfig

config = GenerationConfig(
    max_new_tokens=50,
    temperature=0.1,      # Low for consistency
    top_p=0.9,
    enable_thinking=True, # For Gemini models
    thinking_summary=False
)

# Use with any model
response = model_wrapper.generate(prompt, config)
```

#### Biological Classification
```python
from examples.vision.gemini_vlm_baseline import BiologicalGeminiVLMBaseline

# Specialized for biological organisms
bio_model = BiologicalGeminiVLMBaseline(
    num_classes=len(species_names),
    class_names=species_names,
    model_name="gemini-2.5-pro",
    use_semantic_names=True,
    enable_thinking=True
)
```

## Model Pricing (2025)

### OpenAI Pricing (per million tokens)
- **GPT-4.1**: $2 input / $8 output
- **GPT-4.1 mini**: $0.40 input / $1.60 output
- **GPT-4.1 nano**: $0.10 input / $0.40 output
- **GPT-4o**: Variable pricing
- **GPT-3.5 Turbo**: Lowest cost option

### Gemini Pricing
- **Gemini 2.5 Pro**: Variable based on thinking usage
- **Gemini 2.5 Flash**: Single price tier regardless of input size
- **Gemini 2.5 Flash-Lite**: Lowest cost in the 2.5 family

## Features

### OpenAI Features
- Up to 1 million token context windows (GPT-4.1)
- High-quality vision processing
- Consistent API reliability
- Advanced reasoning models (o3, o4-mini)

### Gemini Features
- **Thinking capabilities**: Models reason through problems step-by-step
- **Thought summaries**: Optional transparency into reasoning process
- **Adjustable thinking budgets**: Balance performance vs cost
- **Native multimodal**: Seamless text and image processing
- **Audio output**: Natural conversational experiences (2.5 models)

## Error Handling

The system includes robust error handling:

```python
try:
    model = model_loader.load_llm("gpt-4o")
except ValueError as e:
    print(f"API key not set: {e}")
except ImportError as e:
    print(f"Install API dependencies: {e}")
```

## Migration from Local Models

Replace existing model names with API equivalents:

```python
# Before (local model)
model = QwenVLBaseline(model_name="Qwen/Qwen2.5-VL-3B-Instruct")

# After (API model)
model = APIVLMBaseline(model_name="gpt-4o")
# or
model = GeminiVLMBaseline(model_name="gemini-2.5-flash")
```

## Best Practices

1. **Choose the right model**:
   - GPT-4.1 nano/mini for cost-effective tasks
   - GPT-4.1/Gemini 2.5 Pro for complex reasoning
   - Gemini Flash models for balanced performance

2. **Use thinking capabilities**:
   - Enable for complex classification tasks
   - Disable for simple, fast classifications

3. **Monitor costs**:
   - Use shorter `max_new_tokens` for classification
   - Choose appropriate model tiers

4. **Handle rate limits**:
   - Built-in retry logic handles temporary failures
   - Monitor API usage in respective dashboards

## Example: Complete Vision Classification

```python
import os
from examples.vision.api_vlm_baseline import APIVLMBaseline

# Set up
os.environ["OPENAI_API_KEY"] = "your-key"

# Create model
model = APIVLMBaseline(
    num_classes=10,
    class_names=["airplane", "automobile", "bird", "cat", "deer", 
                "dog", "frog", "horse", "ship", "truck"],
    model_name="gpt-4o",
    use_semantic_names=True
)

# Train (no actual training for API models)
model.fit(train_loader, train_labels)

# Evaluate
results = model.evaluate(
    test_images, 
    test_labels,
    save_raw_responses=True,
    output_dir="./results",
    benchmark_name="cifar10"
)

print(f"Accuracy: {results['accuracy']:.3f}")
```

This provides access to the most advanced AI models available while maintaining the same simple interface as local models.

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
