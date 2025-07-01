⚠️ **Note to Reviewers**: This project was originally named CLAM. We have not yet refactored all code references to the new name MARVIS.

# MARVIS: Modality Adaptive Reasoning over VISualizations

**MARVIS** is a powerful framework for multi-modal classification that leverages Vision Language Models (VLMs) to perform classification on tabular, audio, and vision data through intelligent visualization and embedding techniques.

## 🚀 Quick Install

```bash
pip install -e ".[vision,audio,api]"
```

## 🌟 Key Features

* **Multi-modal Support**: Tabular, audio, and vision data classification
* **Vision Language Models**: Leverages state-of-the-art VLMs for intelligent reasoning  
* **Advanced Visualizations**: t-SNE, PCA, UMAP, and multi-visualization frameworks
* **API Integration**: Support for OpenAI, Google Gemini, and local models
* **Rich Embeddings**: TabPFN, Whisper, DINOV2, and more

## 💡 Quick Start

### Tabular Data (30 seconds)
```python
from clam.models.clam_tsne import ClamTsneClassifier
from sklearn.datasets import make_classification

# Create sample data
X, y = make_classification(n_samples=100, n_features=10, n_classes=3)

# Create and train classifier
classifier = ClamTsneClassifier(modality="tabular")
classifier.fit(X, y)

# Make predictions
predictions = classifier.predict(X)
print(f"Accuracy: {(predictions == y).mean():.2f}")
```

### Vision Classification
```bash
# Test CIFAR-10 with advanced features
python examples/vision/evaluate_all_vision.py \
    --datasets cifar10 \
    --models clam_tsne \
    --use_3d \
    --use_knn_connections
```

### Audio Classification
```bash
# Test ESC-50 and RAVDESS datasets
python examples/audio/evaluate_all_audio.py \
    --datasets esc50 ravdess \
    --models clam_tsne
```

### API Models
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

classifier = ClamTsneClassifier(
    modality="vision",
    api_model="gpt-4o",
    enable_thinking=True
)
```

## 📚 Documentation

**Full documentation**: [Documentation URL] (coming soon)

* **Installation Guide**: Detailed setup instructions for all modalities
* **User Guides**: Comprehensive guides for vision, audio, and tabular data
* **API Reference**: Complete API documentation with examples
* **Tutorials**: Step-by-step tutorials and best practices

## 🔧 Examples

| Modality | Example | Description |
|----------|---------|-------------|
| **Vision** | `examples/vision/evaluate_all_vision.py` | CIFAR-10/100, ImageNet classification |
| **Audio** | `examples/audio/evaluate_all_audio.py` | ESC-50, RAVDESS classification |
| **Tabular** | `examples/tabular/evaluate_llm_baselines_tabular.py` | OpenML datasets, UCI repository |
| **Multi-modal** | `examples/unified_clam_example.py` | Cross-modality examples |
| **Notebooks** | `notebooks/Getting_Started.ipynb` | Interactive tutorials |

## 🏗️ Advanced Features

* **Multi-Visualization Framework**: Compare multiple embedding methods (PCA, t-SNE, UMAP, Spectral)
* **3D Visualizations**: Interactive 3D t-SNE plots with KNN connections
* **Resource Management**: Intelligent caching and memory optimization
* **Vector Quantization**: Reduce memory footprint for large-scale datasets
* **Evaluation Frameworks**: Comprehensive benchmarking on OpenML CC18 suite

## 🖥️ Platform Support

MARVIS automatically detects and optimizes for your hardware:

### Apple Silicon (M1/M2/M3/M4)
* **Automatic MPS Detection**: Uses Metal Performance Shaders for GPU acceleration
* **Transformers Backend**: Full MPS support for efficient inference
* **Usage**: Set `export VLLM_AVAILABLE=false` to force transformers backend

### NVIDIA GPUs
* **VLLM Support**: Fastest inference with VLLM backend
* **CUDA Optimization**: Automatic device selection and memory management
* **Multi-GPU**: Support for tensor parallelism

### CPU Fallback
* **Universal Support**: Works on any system without GPU
* **Optimized Settings**: Automatic configuration for CPU-only inference

```python
# Device is automatically detected
classifier = ClamTsneClassifier(device="auto")  # Uses MPS on Mac, CUDA on Linux/Windows

# Or specify explicitly
classifier = ClamTsneClassifier(device="mps")   # Force MPS
classifier = ClamTsneClassifier(device="cuda")  # Force CUDA
classifier = ClamTsneClassifier(device="cpu")   # Force CPU
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

* **Repository**: [Repository URL]
* **Issues**: [Issues URL]
* **Documentation**: https://clam.readthedocs.io (coming soon)

---

*MARVIS enables researchers and practitioners to easily apply cutting-edge VLM capabilities to their classification tasks across any data modality.*