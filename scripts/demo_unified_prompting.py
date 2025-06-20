#!/usr/bin/env python
"""
Demonstration of the unified VLM prompting system.

This script shows how the enhanced VLM prompting utilities seamlessly handle
both single and multi-visualization scenarios while maintaining consistent
response formats and including all advanced features.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from clam.utils.vlm_prompting import create_classification_prompt, create_regression_prompt


def demo_single_viz_prompts():
    """Demonstrate single visualization prompts (legacy mode)."""
    print("=" * 60)
    print("SINGLE VISUALIZATION PROMPTS (Legacy Mode)")
    print("=" * 60)
    
    # Classification with advanced features
    class_names = ["Class 0", "Class 1", "Class 2"]
    prompt = create_classification_prompt(
        class_names=class_names,
        modality="tabular",
        use_knn=True,
        knn_k=5,
        use_3d=True,
        legend_text="Training: colored circles, Test: gray squares, Query: red star",
        dataset_description="Iris flower classification dataset with 3 species"
    )
    
    print("Classification Prompt (with KNN + 3D):")
    print("-" * 40)
    print(prompt[:400] + "...")
    print()
    
    # Regression
    target_stats = {'min': 0.5, 'max': 8.5, 'mean': 4.2, 'std': 1.8}
    prompt = create_regression_prompt(
        target_stats=target_stats,
        modality="tabular",
        use_knn=True,
        knn_k=5,
        dataset_description="House price prediction dataset"
    )
    
    print("Regression Prompt (with KNN):")
    print("-" * 40)
    print(prompt[:400] + "...")
    print()


def demo_multi_viz_prompts():
    """Demonstrate multi-visualization prompts (new mode)."""
    print("=" * 60)
    print("MULTI-VISUALIZATION PROMPTS (New Mode)")
    print("=" * 60)
    
    # Multi-viz info
    multi_viz_info = [
        {'method': 'PCA', 'description': 'Principal Component Analysis'},
        {'method': 'TSNE', 'description': 't-SNE visualization'},
        {'method': 'UMAP', 'description': 'UMAP visualization'},
        {'method': 'Spectral', 'description': 'Spectral Embedding'}
    ]
    
    # Classification
    class_names = ["Class 0", "Class 1", "Class 2"]
    prompt = create_classification_prompt(
        class_names=class_names,
        modality="tabular",
        dataset_description="Multi-perspective analysis of classification data",
        multi_viz_info=multi_viz_info
    )
    
    print("Multi-Viz Classification Prompt:")
    print("-" * 40)
    print(prompt[:500] + "...")
    print()
    
    # Regression
    target_stats = {'min': 0.0, 'max': 10.0, 'mean': 5.0, 'std': 2.5}
    prompt = create_regression_prompt(
        target_stats=target_stats,
        modality="tabular",
        dataset_description="Multi-perspective regression analysis",
        multi_viz_info=multi_viz_info
    )
    
    print("Multi-Viz Regression Prompt:")
    print("-" * 40)
    print(prompt[:500] + "...")
    print()


def demo_consistent_features():
    """Demonstrate that all features work consistently across modes."""
    print("=" * 60)
    print("CONSISTENT FEATURES ACROSS MODES")
    print("=" * 60)
    
    # Semantic names work in both modes
    semantic_names = ["Setosa", "Versicolor", "Virginica"]
    
    # Single mode with semantic names
    single_prompt = create_classification_prompt(
        class_names=semantic_names,
        modality="tabular",
        use_semantic_names=True,
        dataset_description="Iris dataset with semantic class names"
    )
    
    # Multi mode with semantic names
    multi_viz_info = [{'method': 'PCA'}, {'method': 'TSNE'}]
    multi_prompt = create_classification_prompt(
        class_names=semantic_names,
        modality="tabular",
        use_semantic_names=True,
        dataset_description="Iris dataset with semantic class names",
        multi_viz_info=multi_viz_info
    )
    
    print("Feature: Semantic Class Names")
    print("-" * 40)
    print("Single mode includes:", "Setosa" in single_prompt, "Versicolor" in single_prompt)
    print("Multi mode includes:", "Setosa" in multi_prompt, "Versicolor" in multi_prompt)
    print()
    
    print("Feature: Structured Response Format")
    print("-" * 40)
    response_format = 'Format your response as: "Class:'
    print("Single mode has format:", response_format in single_prompt)
    print("Multi mode has format:", response_format in multi_prompt)
    print()
    
    print("Feature: Dataset Description")
    print("-" * 40)
    dataset_desc = "Iris dataset with semantic class names"
    print("Single mode includes description:", dataset_desc in single_prompt)
    print("Multi mode includes description:", dataset_desc in multi_prompt)
    print()


def demo_method_specific_guidance():
    """Demonstrate method-specific guidance in multi-viz prompts."""
    print("=" * 60)
    print("METHOD-SPECIFIC GUIDANCE")
    print("=" * 60)
    
    multi_viz_info = [
        {'method': 'PCA', 'description': 'Linear dimensionality reduction'},
        {'method': 'TSNE', 'description': 'Local structure preservation'},
        {'method': 'UMAP', 'description': 'Global and local structure'}
    ]
    
    prompt = create_classification_prompt(
        class_names=["Class 0", "Class 1", "Class 2"],
        modality="tabular",
        multi_viz_info=multi_viz_info
    )
    
    print("Method-Specific Descriptions:")
    print("-" * 40)
    print("PCA guidance:", "linear relationships and directions of maximum variance" in prompt)
    print("t-SNE guidance:", "local neighborhood structures" in prompt) 
    print("UMAP guidance:", "local and global structure" in prompt)
    print()
    
    print("Cross-Method Comparisons:")
    print("-" * 40)
    print("Linear vs nonlinear:", "linear methods (PCA) and nonlinear methods" in prompt)
    print("Local vs global:", "local structure methods" in prompt and "global structure methods" in prompt)
    print()


def main():
    """Run the unified prompting demonstration."""
    print("🔧 UNIFIED VLM PROMPTING SYSTEM DEMONSTRATION")
    print("This demo shows how the enhanced VLM utilities seamlessly support")
    print("both single and multi-visualization scenarios with consistent features.")
    print()
    
    demo_single_viz_prompts()
    demo_multi_viz_prompts()
    demo_consistent_features()
    demo_method_specific_guidance()
    
    print("=" * 60)
    print("✅ INTEGRATION COMPLETE")
    print("=" * 60)
    print("Key achievements:")
    print("• Single source of truth for all VLM prompting")
    print("• Backward compatibility with existing features")
    print("• Seamless multi-visualization support")
    print("• Consistent structured response formats")
    print("• Advanced features (KNN, 3D, semantic names) preserved")
    print("• Method-specific reasoning guidance")
    print("• Cross-visualization comparison insights")


if __name__ == "__main__":
    main()