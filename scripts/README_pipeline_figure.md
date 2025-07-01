# CLAM Pipeline Figure Generator

## Overview

The `generate_clam_pipeline_figure.py` script creates a professional visualization of the CLAM (Classification with Large Autolabeling Models) method pipeline.

## Features

- **Icon-based pipeline**: Four main steps represented with custom geometric icons
- **Gradient arrows**: Beautiful gradient-colored arrows connecting pipeline stages
- **Multimodal support**: Shows support for tabular, audio, and vision data
- **Technical details**: Includes specific model names and methods used
- **High-quality output**: Generates both PNG (300 DPI) and PDF versions

## Pipeline Steps

1. **Multimodal Data**: Input of tabular, audio, and vision datasets
2. **Embedding Generation**: Domain-specific encoders (TabPFN, Whisper, DINOV2)
3. **Dimensionality Reduction**: Visualization techniques (t-SNE, PCA, UMAP)
4. **VLM Classification**: Vision Language Model reasoning and prediction

## Usage

```bash
cd /path/to/clam
conda activate llata
python scripts/generate_clam_pipeline_figure.py
```

## Output

The script generates:
- `data/figures/clam_pipeline_overview.png` - High-resolution PNG (300 DPI)
- `data/figures/clam_pipeline_overview.pdf` - Vector PDF format

## Dependencies

- matplotlib
- numpy
- pathlib (standard library)

The script uses geometric shapes and text instead of emojis to ensure compatibility across all systems and fonts.