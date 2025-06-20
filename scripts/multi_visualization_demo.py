#!/usr/bin/env python
"""
Multi-Visualization Reasoning Demo

This script demonstrates the enhanced CLAM visualization system with
multiple visualization methods composed together for richer VLM reasoning.

Usage:
    python examples/tabular/multi_visualization_demo.py --dataset_name adult
"""

import os
import sys
import argparse
import numpy as np
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from clam.viz import ContextComposer, VisualizationConfig
from clam.viz.context.layouts import LayoutStrategy
from clam.data.dataset import load_tabular_dataset
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_dataset(n_samples=300, n_features=10, n_classes=3, random_state=42):
    """Create a synthetic dataset for demonstration."""
    np.random.seed(random_state)
    
    # Create clustered data
    cluster_centers = np.random.randn(n_classes, n_features) * 3
    
    X = []
    y = []
    
    samples_per_class = n_samples // n_classes
    
    for i in range(n_classes):
        # Generate samples around cluster center
        cluster_samples = np.random.multivariate_normal(
            cluster_centers[i], 
            np.eye(n_features) * 0.5, 
            samples_per_class
        )
        X.append(cluster_samples)
        y.extend([i] * samples_per_class)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Add some noise features
    noise_features = np.random.randn(len(X), n_features // 2) * 0.1
    X = np.hstack([X, noise_features])
    
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Multi-Visualization Reasoning Demo")
    parser.add_argument("--dataset_name", type=str, default="synthetic", 
                       help="Dataset name (synthetic, adult, etc.)")
    parser.add_argument("--n_samples", type=int, default=300,
                       help="Number of samples for synthetic dataset")
    parser.add_argument("--output_dir", type=str, default="./viz_demo_outputs",
                       help="Output directory for visualizations")
    parser.add_argument("--layout_strategy", type=str, default="adaptive_grid",
                       choices=["grid", "adaptive_grid", "sequential", "hierarchical", "focus_plus_context"],
                       help="Layout strategy for composition")
    parser.add_argument("--use_3d", action="store_true",
                       help="Use 3D visualizations where supported")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Multi-Visualization Reasoning Demo")
    logger.info("=" * 50)
    
    # Load or create dataset
    if args.dataset_name == "synthetic":
        logger.info("Creating synthetic dataset...")
        X, y = create_sample_dataset(
            n_samples=args.n_samples,
            random_state=args.seed
        )
        dataset_info = {
            'name': 'synthetic',
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y))
        }
    else:
        logger.info(f"Loading dataset: {args.dataset_name}")
        try:
            # Try to load from CLAM dataset utilities
            dataset = load_tabular_dataset(args.dataset_name)
            X, y = dataset['X'], dataset['y']
            dataset_info = dataset
        except:
            logger.error(f"Could not load dataset: {args.dataset_name}")
            logger.info("Using synthetic dataset instead...")
            X, y = create_sample_dataset(random_state=args.seed)
            dataset_info = {'name': 'synthetic_fallback'}
    
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    
    logger.info(f"Split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Create context composer
    from clam.viz.context.composer import CompositionConfig
    
    config = CompositionConfig(
        layout_strategy=LayoutStrategy[args.layout_strategy.upper()],
        reasoning_focus="comparison",
        optimize_for_vlm=True
    )
    
    composer = ContextComposer(config)
    
    # Add multiple visualizations
    logger.info("Adding visualization methods...")
    
    # 1. PCA - Linear baseline
    composer.add_visualization(
        'pca',
        config={'whiten': False},
        viz_config=VisualizationConfig(
            use_3d=args.use_3d,
            title="PCA - Linear Structure",
            random_state=args.seed
        )
    )
    
    # 2. t-SNE - Local structure
    composer.add_visualization(
        'tsne',
        config={'perplexity': min(30, len(X_train) // 4), 'max_iter': 500},
        viz_config=VisualizationConfig(
            use_3d=args.use_3d,
            title="t-SNE - Local Clusters",
            random_state=args.seed
        )
    )
    
    # 3. UMAP - Global + Local structure
    try:
        composer.add_visualization(
            'umap',
            config={'n_neighbors': 15, 'min_dist': 0.1},
            viz_config=VisualizationConfig(
                use_3d=args.use_3d,
                title="UMAP - Global Structure",
                random_state=args.seed
            )
        )
    except ImportError:
        logger.warning("UMAP not available, skipping")
    
    # 4. Spectral Embedding - Graph structure
    composer.add_visualization(
        'spectral',
        config={'n_neighbors': max(2, len(X_train) // 20), 'affinity': 'nearest_neighbors'},
        viz_config=VisualizationConfig(
            use_3d=args.use_3d,
            title="Spectral - Graph Structure",
            random_state=args.seed
        )
    )
    
    # 5. Isomap - Geodesic distances
    composer.add_visualization(
        'isomap',
        config={'n_neighbors': 10},
        viz_config=VisualizationConfig(
            use_3d=args.use_3d,
            title="Isomap - Geodesic Distances",
            random_state=args.seed
        )
    )
    
    logger.info(f"Added {len(composer.visualizations)} visualization methods")
    
    # Fit all visualizations
    logger.info("Fitting visualization methods...")
    composer.fit(X_train, y_train, X_test)
    
    # Create composed visualization
    logger.info("Creating composed visualization...")
    
    # Highlight a few interesting points
    highlight_indices = [0, len(X_train)//4, len(X_train)//2, -1]
    
    composed_image = composer.compose_layout(
        highlight_indices=highlight_indices,
        layout_strategy=LayoutStrategy[args.layout_strategy.upper()]
    )
    
    # Save composed image
    output_path = os.path.join(args.output_dir, f"composed_visualization_{args.layout_strategy}.png")
    composed_image.save(output_path)
    logger.info(f"Saved composed visualization: {output_path}")
    
    # Generate reasoning prompt
    logger.info("Generating reasoning prompt...")
    
    reasoning_prompt = composer.generate_reasoning_prompt(
        highlight_indices=highlight_indices,
        custom_context=f"This is a {dataset_info.get('name', 'tabular')} dataset with "
                      f"{len(np.unique(y))} classes. The highlighted points represent "
                      f"examples from different parts of the dataset.",
        task_description="Analyze the consistency of cluster structure across different "
                        "visualization methods and identify which patterns are most reliable."
    )
    
    # Save reasoning prompt
    prompt_path = os.path.join(args.output_dir, "reasoning_prompt.txt")
    with open(prompt_path, 'w') as f:
        f.write(reasoning_prompt)
    logger.info(f"Saved reasoning prompt: {prompt_path}")
    
    # Get visualization comparison
    comparison = composer.get_visualization_comparison()
    
    # Save comparison
    import json
    comparison_path = os.path.join(args.output_dir, "visualization_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info(f"Saved visualization comparison: {comparison_path}")
    
    # Perform multi-visualization reasoning
    logger.info("Performing multi-visualization reasoning...")
    
    reasoning_result = composer.reason_over_data(
        X_train, y_train,
        reasoning_chain=[
            "Which visualizations show the clearest cluster separation?",
            "Are there any outliers that appear consistently across methods?", 
            "How do linear (PCA) and nonlinear methods differ in their representation?",
            "Which patterns can we be most confident about?"
        ],
        highlight_indices=highlight_indices
    )
    
    # Save reasoning result metadata
    reasoning_metadata_path = os.path.join(args.output_dir, "reasoning_metadata.json")
    with open(reasoning_metadata_path, 'w') as f:
        # Convert PIL image to filename reference for JSON serialization
        metadata = reasoning_result['metadata'].copy()
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved reasoning metadata: {reasoning_metadata_path}")
    
    # Display summary
    print("\n" + "="*60)
    print("MULTI-VISUALIZATION ANALYSIS SUMMARY")
    print("="*60)
    print(f"Dataset: {dataset_info.get('name', 'unknown')}")
    print(f"Samples: {len(X_train)} train, {len(X_test)} test")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Visualizations: {len(composer.visualizations)}")
    print(f"Layout Strategy: {args.layout_strategy}")
    print(f"3D Mode: {args.use_3d}")
    print(f"Output Directory: {args.output_dir}")
    
    print(f"\nGenerated Files:")
    print(f"  - Composed visualization: {output_path}")
    print(f"  - Reasoning prompt: {prompt_path}")
    print(f"  - Visualization comparison: {comparison_path}")
    print(f"  - Reasoning metadata: {reasoning_metadata_path}")
    
    print(f"\nVisualization Methods:")
    for viz in composer.visualizations:
        supports_new = "✓" if viz.supports_new_data else "✗"
        supports_3d = "✓" if viz.supports_3d else "✗"
        print(f"  - {viz.method_name:20s} (New data: {supports_new}, 3D: {supports_3d})")
    
    if comparison.get('recommendations'):
        print(f"\nRecommendations:")
        for rec in comparison['recommendations']:
            print(f"  - {rec}")
    
    print(f"\n🎉 Multi-visualization analysis complete!")
    print(f"The composed visualization and reasoning prompt are ready for VLM analysis.")


if __name__ == "__main__":
    main()