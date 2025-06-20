CLAM T-SNE Enhanced Reasoning Plan                                                                             │ │
│ │                                                                                                                │ │
│ │ Overview                                                                                                       │ │
│ │                                                                                                                │ │
│ │ Transform the current clam_tsne model into a flexible, multi-visualization reasoning system with a composable  │ │
│ │ context framework for VLM backends.                                                                            │ │
│ │                                                                                                                │ │
│ │ 1. Core Architecture Design                                                                                    │ │
│ │                                                                                                                │ │
│ │ 1.1 Visualization Module Abstraction (clam/viz/)                                                               │ │
│ │                                                                                                                │ │
│ │ clam/viz/                                                                                                      │ │
│ │ ├── __init__.py                                                                                                │ │
│ │ ├── base.py                 # Abstract base classes                                                            │ │
│ │ ├── embeddings/             # Dimensionality reduction visualizations                                          │ │
│ │ │   ├── tsne.py            # Current T-SNE implementation                                                      │ │
│ │ │   ├── umap.py            # UMAP implementations                                                              │ │
│ │ │   ├── manifold.py        # sklearn.manifold wrappers                                                         │ │
│ │ │   └── pca.py             # Enhanced PCA                                                                      │ │
│ │ ├── decision/               # Decision boundary visualizations                                                 │ │
│ │ │   ├── regions.py         # mlxtend decision regions                                                          │ │
│ │ │   └── boundaries.py      # Custom decision boundaries                                                        │ │
│ │ ├── patterns/               # Pattern analysis visualizations                                                  │ │
│ │ │   ├── frequent.py        # mlxtend frequent patterns                                                         │ │
│ │ │   └── associations.py    # Association rule mining                                                           │ │
│ │ └── context/               # Context composition system                                                        │ │
│ │     ├── composer.py        # Main context composer                                                             │ │
│ │     ├── layouts.py         # Multi-viz layout strategies                                                       │ │
│ │     └── prompts.py         # Text prompt generation                                                            │ │
│ │                                                                                                                │ │
│ │ 1.2 Context Composer System                                                                                    │ │
│ │                                                                                                                │ │
│ │ - Modular Design: Each visualization as an independent component                                               │ │
│ │ - Chaining Support: Multiple visualizations in single VLM prompt                                               │ │
│ │ - Layout Management: Grid, sequential, and hierarchical layouts                                                │ │
│ │ - Prompt Integration: Automatic text generation for each visualization                                         │ │
│ │                                                                                                                │ │
│ │ 2. Implementation Phases                                                                                       │ │
│ │                                                                                                                │ │
│ │ Phase 1: Foundation (Weeks 1-2)                                                                                │ │
│ │                                                                                                                │ │
│ │ - Create abstract base classes for visualizations                                                              │ │
│ │ - Implement context composer framework                                                                         │ │
│ │ - Migrate existing T-SNE functionality                                                                         │ │
│ │ - Add UMAP support with scikit-learn interface                                                                 │ │
│ │                                                                                                                │ │
│ │ Phase 2: Core Visualizations (Weeks 3-4)                                                                       │ │
│ │                                                                                                                │ │
│ │ - Implement all sklearn.manifold methods:                                                                      │ │
│ │   - LocallyLinearEmbedding (standard, modified, Hessian, LTSA)                                                 │ │
│ │   - SpectralEmbedding                                                                                          │ │
│ │   - Isomap                                                                                                     │ │
│ │   - MDS                                                                                                        │ │
│ │ - Add mlxtend decision regions                                                                                 │ │
│ │ - Create layout management system                                                                              │ │
│ │                                                                                                                │ │
│ │ Phase 3: Advanced Features (Weeks 5-6)                                                                         │ │
│ │                                                                                                                │ │
│ │ - Implement mlxtend frequent patterns                                                                          │ │
│ │ - Add multi-visualization chaining                                                                             │ │
│ │ - Create intelligent prompt generation                                                                         │ │
│ │ - Optimize for VLM consumption                                                                                 │ │
│ │                                                                                                                │ │
│ │ Phase 4: Integration & Testing (Weeks 7-8)                                                                     │ │
│ │                                                                                                                │ │
│ │ - Integrate with existing CLAM pipeline                                                                        │ │
│ │ - Create comprehensive tests                                                                                   │ │
│ │ - Performance optimization                                                                                     │ │
│ │ - Documentation and examples                                                                                   │ │
│ │                                                                                                                │ │
│ │ 3. Technical Specifications                                                                                    │ │
│ │                                                                                                                │ │
│ │ 3.1 Base Visualization Interface                                                                               │ │
│ │                                                                                                                │ │
│ │ class BaseVisualization(ABC):                                                                                  │ │
│ │     @abstractmethod                                                                                            │ │
│ │     def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray                                        │ │
│ │                                                                                                                │ │
│ │     @abstractmethod                                                                                            │ │
│ │     def generate_plot(self, **kwargs) -> PIL.Image                                                             │ │
│ │                                                                                                                │ │
│ │     @abstractmethod                                                                                            │ │
│ │     def get_description(self) -> str                                                                           │ │
│ │                                                                                                                │ │
│ │ 3.2 Context Composer API                                                                                       │ │
│ │                                                                                                                │ │
│ │ class ContextComposer:                                                                                         │ │
│ │     def add_visualization(self, viz_type: str, config: dict)                                                   │ │
│ │     def compose_layout(self, layout_strategy: str) -> PIL.Image                                                │ │
│ │     def generate_prompt(self, dataset_context: dict) -> str                                                    │ │
│ │     def reason_over_data(self, X, y, reasoning_chain: List[str])                                               │ │
│ │                                                                                                                │ │
│ │ 3.3 Supported Visualization Types                                                                              │ │
│ │                                                                                                                │ │
│ │ - tsne: Enhanced T-SNE with 2D/3D, multiple perplexities                                                       │ │
│ │ - umap: UMAP with various distance metrics and parameters                                                      │ │
│ │ - lle: LocallyLinearEmbedding variants (standard, modified, hessian, ltsa)                                     │ │
│ │ - spectral: SpectralEmbedding with different affinity matrices                                                 │ │
│ │ - isomap: Isomap with geodesic distance preservation                                                           │ │
│ │ - mds: Multidimensional Scaling (metric and non-metric)                                                        │ │
│ │ - decision_regions: mlxtend decision boundary visualization                                                    │ │
│ │ - frequent_patterns: Pattern mining visualizations                                                             │ │
│ │                                                                                                                │ │
│ │ 4. Key Features                                                                                                │ │
│ │                                                                                                                │ │
│ │ 4.1 Multi-Visualization Reasoning                                                                              │ │
│ │                                                                                                                │ │
│ │ - Chain multiple visualizations to show different data perspectives                                            │ │
│ │ - Automatic prompt generation explaining each visualization's insights                                         │ │
│ │ - Cross-reference patterns between different embedding spaces                                                  │ │
│ │                                                                                                                │ │
│ │ 4.2 Adaptive Context Generation                                                                                │ │
│ │                                                                                                                │ │
│ │ - Dataset-aware prompt customization                                                                           │ │
│ │ - Visualization-specific reasoning prompts                                                                     │ │
│ │ - Confidence and uncertainty communication to VLM                                                              │ │
│ │                                                                                                                │ │
│ │ 4.3 Performance Optimization                                                                                   │ │
│ │                                                                                                                │ │
│ │ - Lazy evaluation of expensive computations                                                                    │ │
│ │ - Caching of embeddings and visualizations                                                                     │ │
│ │ - GPU acceleration where available (UMAP, some sklearn methods)                                                │ │
│ │                                                                                                                │ │
│ │ 5. Integration Points                                                                                          │ │
│ │                                                                                                                │ │
│ │ 5.1 Existing CLAM Pipeline                                                                                     │ │
│ │                                                                                                                │ │
│ │ - Maintain backward compatibility with current clam_tsne                                                       │ │
│ │ - Integrate with model_loader and VLM backends                                                                 │ │
│ │ - Support both VLLM and transformers backends                                                                  │ │
│ │                                                                                                                │ │
│ │ 5.2 Examples Integration                                                                                       │ │
│ │                                                                                                                │ │
│ │ - Update examples/tabular/ to showcase new capabilities                                                        │ │
│ │ - Create demonstration notebooks                                                                               │ │
│ │ - Add to existing evaluation pipelines                                                                         │ │
│ │                                                                                                                │ │
│ │ 6. Expected Benefits                                                                                           │ │
│ │                                                                                                                │ │
│ │ 6.1 Enhanced Reasoning Capability                                                                              │ │
│ │                                                                                                                │ │
│ │ - Multiple perspectives on the same data                                                                       │ │
│ │ - Richer context for VLM decision-making                                                                       │ │
│ │ - Better handling of complex, high-dimensional datasets                                                        │ │
│ │                                                                                                                │ │
│ │ 6.2 Research Flexibility                                                                                       │ │
│ │                                                                                                                │ │
│ │ - Easy experimentation with different embedding methods                                                        │ │
│ │ - Rapid prototyping of new visualization combinations                                                          │ │
│ │ - Systematic evaluation of reasoning approaches                                                                │ │
│ │                                                                                                                │ │
│ │ 6.3 Practical Applications                                                                                     │ │
│ │                                                                                                                │ │
│ │ - Better performance on challenging tabular datasets                                                           │ │
│ │ - More interpretable model decisions                                                                           │ │
│ │ - Improved few-shot learning through richer context                                                            │ │
│ │                                                                                                                │ │
│ │ 7. Success Metrics                                                                                             │ │
│ │                                                                                                                │ │
│ │ - Maintain or improve accuracy on existing benchmarks                                                          │ │
│ │ - Demonstrate improved performance on complex datasets                                                         │ │
│ │ - Show enhanced interpretability through multi-viz reasoning                                                   │ │
│ │ - Achieve sub-linear scaling with number of visualizations     

Sphinx Documentation Plan for CLAM

Current State Analysis

- Main README: 606 lines, comprehensive but unwieldy
- Scattered docs: Multiple README files in subdirectories (audio/, tabular/openml_cc18/, etc.)
- Rich content: Detailed usage examples, API guides, troubleshooting
- No Sphinx setup: No existing conf.py or docs/ directory
- Modern structure: Uses pyproject.toml for packaging

1. Sphinx Setup & Configuration

docs/
├── conf.py                    # Sphinx configuration
├── index.rst                  # Main documentation index
├── requirements.txt           # Documentation dependencies
├── _static/                   # Custom CSS, images, etc.
├── _templates/                # Custom Sphinx templates
└── make.bat / Makefile        # Build automation

Key Sphinx Extensions:
- sphinx.ext.autodoc - Auto-generate API docs from docstrings
- sphinx.ext.napoleon - Google/NumPy style docstrings
- sphinx.ext.viewcode - Source code links
- sphinx.ext.intersphinx - Cross-reference external docs
- sphinx.ext.autosummary - Generate summary tables
- myst_parser - Markdown support
- sphinx_rtd_theme - ReadTheDocs theme

2. Documentation Structure

docs/
├── index.rst
├── getting-started/
│   ├── installation.rst
│   ├── quick-start.rst
│   └── configuration.rst
├── user-guide/
│   ├── vision/
│   │   ├── cifar-classification.rst
│   │   ├── api-models.rst
│   │   └── advanced-features.rst
│   ├── audio/
│   │   ├── esc50-ravdess.rst
│   │   ├── baselines.rst
│   │   └── visualization.rst
│   ├── tabular/
│   │   ├── openml-datasets.rst
│   │   ├── llm-baselines.rst
│   │   └── regression-support.rst
│   └── api-models/
│       ├── openai-integration.rst
│       ├── gemini-integration.rst
│       └── pricing-guide.rst
├── tutorials/
│   ├── basic-classification.rst
│   ├── multi-modal-pipeline.rst
│   └── custom-datasets.rst
├── api-reference/
│   ├── clam.data.rst
│   ├── clam.models.rst
│   ├── clam.train.rst
│   └── clam.utils.rst
├── technical-guides/
│   ├── resource-management.rst
│   ├── caching-system.rst
│   ├── vector-quantization.rst
│   └── evaluation-frameworks.rst
├── examples/
│   ├── vision-examples.rst
│   ├── audio-examples.rst
│   └── tabular-examples.rst
├── troubleshooting.rst
├── changelog.rst
└── contributing.rst

3. README Cleanup Strategy

New README structure (reduce from 606 to ~150 lines):

# CLAM: CLassify Anything Model

Brief description and key features

## Quick Install
pip install -e ".[vision,audio,api]"

## Quick Start
# 30-second example for each modality

## Documentation
Full documentation: https://clam.readthedocs.io

## Examples
Link to examples directory with one-liner for each

## Contributing & License
Brief section with links to detailed docs

What moves to Sphinx docs:
- Detailed installation instructions → getting-started/installation.rst
- All API model documentation → user-guide/api-models/
- Advanced usage examples → tutorials/
- Vector quantization details → technical-guides/vector-quantization.rst
- Troubleshooting → troubleshooting.rst

4. Migration Strategy

Phase 1: Setup Infrastructure
- Create Sphinx configuration
- Set up ReadTheDocs or GitHub Pages
- Configure automated builds

Phase 2: Content Migration
- Move existing README sections to appropriate .rst files
- Convert Markdown to reStructuredText where needed
- Organize scattered README files into unified structure

Phase 3: API Documentation
- Add comprehensive docstrings to code
- Set up autodoc for API reference
- Create cross-references between user guide and API docs

Phase 4: Enhancement
- Add tutorials with executable examples
- Create interactive Jupyter notebooks
- Add search functionality and better navigation

5. Automation & Maintenance

GitHub Actions Workflow:
name: Documentation
on: [push, pull_request]
jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -e ".[docs]"
          pip install -r docs/requirements.txt
      - name: Build docs
        run: |
          cd docs
          make html
      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/_build/html

Benefits of This Approach:

1. Improved Navigation: Structured hierarchy vs. single long README
2. Searchable Documentation: Sphinx provides built-in search
3. API Integration: Autodoc generates API docs from code
4. Better Maintenance: Separate files easier to update than monolithic README
5. Professional Appearance: Standard documentation site layout
6. Cross-References: Link between user guide and API reference
7. Versioned Docs: Support for multiple versions if needed

Implementation Priority:
1. High: Basic Sphinx setup + README cleanup
2. Medium: Content migration + API docs
3. Low: Advanced features + automation

CLAM Codebase Improvement Plan                                                                                 │ │
│ │                                                                                                                │ │
│ │ 1. Unified Results Saving with Resource Manager                                                                │ │
│ │                                                                                                                │ │
│ │ Current State Analysis                                                                                         │ │
│ │                                                                                                                │ │
│ │ - ✅ Good News: A comprehensive ResultsManager already exists in clam/utils/results_manager.py                  │ │
│ │ - ❌ Problem: Many evaluation scripts still use ad-hoc result saving patterns                                   │ │
│ │ - Directory Structure: Results currently scattered across different patterns                                   │ │
│ │                                                                                                                │ │
│ │ Plan: Standardize on Existing Results Manager                                                                  │ │
│ │                                                                                                                │ │
│ │ Phase 1: Audit and Migration (1-2 days)                                                                        │ │
│ │                                                                                                                │ │
│ │ 1. Audit all evaluation scripts for current result saving patterns:                                            │ │
│ │   - examples/tabular/evaluate_llm_baselines_tabular.py - Uses some unified saving                              │ │
│ │   - examples/audio/ scripts - Mixed patterns                                                                   │ │
│ │   - examples/vision/ scripts - Mixed patterns                                                                  │ │
│ │   - Legacy scripts with custom output directories                                                              │ │
│ │ 2. Create migration utilities:                                                                                 │ │
│ │   - scripts/migrate_legacy_results.py - Convert old result formats                                             │ │
│ │   - Detection logic for different legacy formats                                                               │ │
│ │   - Batch migration with progress tracking                                                                     │ │
│ │                                                                                                                │ │
│ │ Phase 2: Standardize All Scripts (2-3 days)                                                                    │ │
│ │                                                                                                                │ │
│ │ 1. Update all evaluation scripts to use unified ResultsManager:                                                │ │
│ │ # Standard pattern for all scripts                                                                             │ │
│ │ from clam.utils import get_results_manager, EvaluationResults, ExperimentMetadata                              │ │
│ │                                                                                                                │ │
│ │ results_manager = get_results_manager()                                                                        │ │
│ │ results_manager.save_evaluation_results(                                                                       │ │
│ │     model_name=model_name,                                                                                     │ │
│ │     dataset_id=dataset_id,                                                                                     │ │
│ │     modality=modality,  # "tabular", "audio", "vision"                                                         │ │
│ │     results=EvaluationResults(...),                                                                            │ │
│ │     experiment_metadata=ExperimentMetadata(...)                                                                │ │
│ │ )                                                                                                              │ │
│ │ 2. Standardized directory structure: ~/.clam/results/<modality>/<dataset_id>/<model_name>/                     │ │
│ │   - results.json - Main metrics                                                                                │ │
│ │   - metadata.json - Full experiment configuration                                                              │ │
│ │   - artifacts/ - Visualizations, plots, etc.                                                                   │ │
│ │                                                                                                                │ │
│ │ Phase 3: Enhanced CLI Tools (1 day)                                                                            │ │
│ │                                                                                                                │ │
│ │ 1. Results management CLI:                                                                                     │ │
│ │ # List and browse results                                                                                      │ │
│ │ clam results list --modality tabular --dataset adult                                                           │ │
│ │ clam results compare --models tabllm,jolt --dataset adult                                                      │ │
│ │ clam results export --format csv --output summary.csv                                                          │ │
│ │                                                                                                                │ │
│ │ # Migration and cleanup                                                                                        │ │
│ │ clam results migrate ./legacy_results_dir                                                                      │ │
│ │ clam results cleanup --older-than 30days                                                                       │ │
│ │ 2. Integration with existing resource manager for consistent path handling                                     │ │
│ │                                                                                                                │ │
│ │ ---                                                                                                            │ │
│ │ 2. Unified CLAM-T-SNE Interface (Scikit-learn Style)                                                           │ │
│ │                                                                                                                │ │
│ │ Current State Analysis                                                                                         │ │
│ │                                                                                                                │ │
│ │ - Modality Support: Tabular (mature), Audio (good), Vision (good)                                              │ │
│ │ - Different Classes: ClamTsneClassifier, ClamAudioTsneClassifier, ClamImageTsneClassifier                      │ │
│ │ - Inconsistent APIs: Different parameter names, methods across modalities                                      │ │
│ │ - Scattered Implementation: Logic split between main models and examples                                       │ │
│ │                                                                                                                │ │
│ │ Plan: Create Unified Scikit-learn Interface                                                                    │ │
│ │                                                                                                                │ │
│ │ Phase 1: Design Unified Interface (2 days)                                                                     │ │
│ │                                                                                                                │ │
│ │ 1. Core Interface Design:                                                                                      │ │
│ │ from clam import ClamClassifier                                                                                │ │
│ │                                                                                                                │ │
│ │ # Unified interface for all modalities                                                                         │ │
│ │ classifier = ClamClassifier(                                                                                   │ │
│ │     modality="tabular",  # "tabular", "audio", "vision"                                                        │ │
│ │     model="clam_tsne",   # Could expand to other models later                                                  │ │
│ │     **modality_specific_params                                                                                 │ │
│ │ )                                                                                                              │ │
│ │                                                                                                                │ │
│ │ # Consistent scikit-learn API                                                                                  │ │
│ │ classifier.fit(X_train, y_train)                                                                               │ │
│ │ predictions = classifier.predict(X_test)                                                                       │ │
│ │ probabilities = classifier.predict_proba(X_test)                                                               │ │
│ │ score = classifier.score(X_test, y_test)                                                                       │ │
│ │ 2. Modality-Specific Dataset Types:                                                                            │ │
│ │ # Type-safe dataset containers                                                                                 │ │
│ │ from clam.data import TabularDataset, AudioDataset, VisionDataset                                              │ │
│ │                                                                                                                │ │
│ │ # Tabular: pandas DataFrame or numpy arrays                                                                    │ │
│ │ tabular_data = TabularDataset(X=df_features, y=df_target)                                                      │ │
│ │                                                                                                                │ │
│ │ # Audio: file paths with metadata                                                                              │ │
│ │ audio_data = AudioDataset(                                                                                     │ │
│ │     paths=["audio1.wav", "audio2.wav"],                                                                        │ │
│ │     labels=["speech", "music"],                                                                                │ │
│ │     metadata={"sample_rate": 16000}                                                                            │ │
│ │ )                                                                                                              │ │
│ │                                                                                                                │ │
│ │ # Vision: file paths or image arrays                                                                           │ │
│ │ vision_data = VisionDataset(                                                                                   │ │
│ │     paths=["img1.jpg", "img2.jpg"],                                                                            │ │
│ │     labels=["cat", "dog"],                                                                                     │ │
│ │     metadata={"image_size": (224, 224)}                                                                        │ │
│ │ )                                                                                                              │ │
│ │                                                                                                                │ │
│ │ Phase 2: Implementation Architecture (3-4 days)                                                                │ │
│ │                                                                                                                │ │
│ │ 1. Base Class Hierarchy:                                                                                       │ │
│ │ # clam/models/base.py                                                                                          │ │
│ │ class ClamBaseEstimator:                                                                                       │ │
│ │     """Base class following scikit-learn conventions"""                                                        │ │
│ │                                                                                                                │ │
│ │ class ClamClassifier(ClamBaseEstimator):                                                                       │ │
│ │     """Unified classifier for all modalities"""                                                                │ │
│ │                                                                                                                │ │
│ │ # Internal dispatch to modality-specific implementations                                                       │ │
│ │ class TabularClamClassifier(ClamBaseEstimator):                                                                │ │
│ │ class AudioClamClassifier(ClamBaseEstimator):                                                                  │ │
│ │ class VisionClamClassifier(ClamBaseEstimator):                                                                 │ │
│ │ 2. Configuration System:                                                                                       │ │
│ │ # Unified parameter structure                                                                                  │ │
│ │ @dataclass                                                                                                     │ │
│ │ class ClamConfig:                                                                                              │ │
│ │     # Common parameters                                                                                        │ │
│ │     vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"                                                             │ │
│ │     embedding_size: int = 1000                                                                                 │ │
│ │     tsne_params: TsneConfig = field(default_factory=TsneConfig)                                                │ │
│ │                                                                                                                │ │
│ │     # Modality-specific configs                                                                                │ │
│ │     tabular_config: TabularConfig = field(default_factory=TabularConfig)                                       │ │
│ │     audio_config: AudioConfig = field(default_factory=AudioConfig)                                             │ │
│ │     vision_config: VisionConfig = field(default_factory=VisionConfig)                                          │ │
│ │ 3. Modality Detection and Validation:                                                                          │ │
│ │ # Automatic modality detection                                                                                 │ │
│ │ def detect_modality(X, y=None):                                                                                │ │
│ │     if isinstance(X, (pd.DataFrame, np.ndarray)):                                                              │ │
│ │         return "tabular"                                                                                       │ │
│ │     elif isinstance(X[0], (str, Path)) and X[0].suffix in AUDIO_EXTENSIONS:                                    │ │
│ │         return "audio"                                                                                         │ │
│ │     elif isinstance(X[0], (str, Path)) and X[0].suffix in IMAGE_EXTENSIONS:                                    │ │
│ │         return "vision"                                                                                        │ │
│ │     else:                                                                                                      │ │
│ │         raise ValueError("Cannot detect modality")                                                             │ │
│ │                                                                                                                │ │
│ │ Phase 3: Enhanced Features (2-3 days)                                                                          │ │
│ │                                                                                                                │ │
│ │ 1. Pipeline Integration:                                                                                       │ │
│ │ from sklearn.pipeline import Pipeline                                                                          │ │
│ │ from clam import ClamClassifier                                                                                │ │
│ │                                                                                                                │ │
│ │ # Work with scikit-learn pipelines                                                                             │ │
│ │ pipeline = Pipeline([                                                                                          │ │
│ │     ('preprocessor', StandardScaler()),  # For tabular                                                         │ │
│ │     ('classifier', ClamClassifier(modality="tabular"))                                                         │ │
│ │ ])                                                                                                             │ │
│ │ 2. Cross-Validation Support:                                                                                   │ │
│ │ from sklearn.model_selection import cross_val_score                                                            │ │
│ │                                                                                                                │ │
│ │ # Native scikit-learn compatibility                                                                            │ │
│ │ scores = cross_val_score(                                                                                      │ │
│ │     ClamClassifier(modality="tabular"),                                                                        │ │
│ │     X, y, cv=5                                                                                                 │ │
│ │ )                                                                                                              │ │
│ │ 3. Hyperparameter Optimization:                                                                                │ │
│ │ from sklearn.model_selection import GridSearchCV                                                               │ │
│ │                                                                                                                │ │
│ │ param_grid = {                                                                                                 │ │
│ │     'embedding_size': [500, 1000, 1500],                                                                       │ │
│ │     'tsne_params__perplexity': [10, 30, 50]                                                                    │ │
│ │ }                                                                                                              │ │
│ │                                                                                                                │ │
│ │ grid_search = GridSearchCV(                                                                                    │ │
│ │     ClamClassifier(modality="tabular"),                                                                        │ │
│ │     param_grid, cv=3                                                                                           │ │
│ │ )                                                                                                              │ │
│ │                                                                                                                │ │
│ │ Phase 4: Documentation and Examples (1-2 days)                                                                 │ │
│ │                                                                                                                │ │
│ │ 1. Comprehensive Documentation:                                                                                │ │
│ │   - API reference with all parameters                                                                          │ │
│ │   - Modality-specific guides                                                                                   │ │
│ │   - Migration guide from current interfaces                                                                    │ │
│ │   - Performance and scaling considerations                                                                     │ │
│ │ 2. Example Notebooks:                                                                                          │ │
│ │ # examples/unified_interface_demo.ipynb                                                                        │ │
│ │                                                                                                                │ │
│ │ # Tabular example                                                                                              │ │
│ │ classifier = ClamClassifier(modality="tabular")                                                                │ │
│ │ classifier.fit(X_train, y_train)                                                                               │ │
│ │ accuracy = classifier.score(X_test, y_test)                                                                    │ │
│ │                                                                                                                │ │
│ │ # Audio example                                                                                                │ │
│ │ audio_classifier = ClamClassifier(                                                                             │ │
│ │     modality="audio",                                                                                          │ │
│ │     audio_config=AudioConfig(embedding_model="whisper")                                                        │ │
│ │ )                                                                                                              │ │
│ │                                                                                                                │ │
│ │ # Vision example                                                                                               │ │
│ │ vision_classifier = ClamClassifier(                                                                            │ │
│ │     modality="vision",                                                                                         │ │
│ │     vision_config=VisionConfig(dinov2_model="dinov2_vitb14")                                                   │ │
│ │ )                                                                                                              │ │
│ │                                                                                                                │ │
│ │ ---                                                                                                            │ │
│ │ Implementation Timeline                                                                                        │ │
│ │                                                                                                                │ │
│ │ Week 1: Results Unification                                                                                    │ │
│ │                                                                                                                │ │
│ │ - Days 1-2: Audit and create migration utilities                                                               │ │
│ │ - Days 3-5: Update all evaluation scripts                                                                      │ │
│ │                                                                                                                │ │
│ │ Week 2: CLAM-T-SNE Interface                                                                                   │ │
│ │                                                                                                                │ │
│ │ - Days 1-2: Design unified interface and dataset types                                                         │ │
│ │ - Days 3-5: Implement base architecture                                                                        │ │
│ │                                                                                                                │ │
│ │ Week 3: Enhancement and Polish                                                                                 │ │
│ │                                                                                                                │ │
│ │ - Days 1-3: Enhanced features and pipeline integration                                                         │ │
│ │ - Days 4-5: Documentation and examples                                                                         │ │
│ │                                                                                                                │ │
│ │ Success Metrics                                                                                                │ │
│ │                                                                                                                │ │
│ │ 1. Results Unification:                                                                                        │ │
│ │   - All evaluation scripts use unified ResultsManager                                                          │ │
│ │   - Consistent directory structure across all modalities                                                       │ │
│ │   - Working migration tools for legacy results                                                                 │ │
│ │ 2. Unified Interface:                                                                                          │ │
│ │   - Single ClamClassifier works across all modalities                                                          │ │
│ │   - Full scikit-learn compatibility (pipelines, cross-validation, etc.)                                        │ │
│ │   - Maintained backward compatibility with existing code                                                       │ │
│ │   - Comprehensive documentation and examples                                                                   │ │
│ │                                                                                                                │ │
│ │ Risk Mitigation                                                                                                │ │
│ │                                                                                                                │ │
│ │ - Backward Compatibility: Keep existing interfaces working during transition                                   │ │
│ │ - Incremental Migration: Update scripts one modality at a time                                                 │ │
│ │ - Comprehensive Testing: Validate all modalities work correctly                                                │ │
│ │ - Documentation: Clear migration guides for existing users 