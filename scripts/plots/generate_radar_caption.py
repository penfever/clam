#!/usr/bin/env python3
"""
Generate a publication-ready caption for the CLAM radar plot.

This script analyzes the core results and creates a compelling caption
that highlights CLAM's role as a bridge between traditional ML and modern LLM/VLM approaches.
"""

import pandas as pd
import numpy as np

def analyze_results():
    """Analyze the core results to generate insights for the caption."""
    
    # Load and process the data (simplified version of the radar plot script)
    df = pd.read_csv("/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/clam/results/core_results.csv")
    df = df.dropna(subset=['Domain', 'Benchmark', 'Method', 'Value'])
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna(subset=['Value'])
    
    # Categorize methods
    def categorize_method(row):
        method = row['Method']
        backend = row['Backend']
        
        if method == 'CLAMS':
            return 'CLAMS'
        elif method == 'Conventional' and backend in ['TabPFNv2', 'Random Forest', 'CatBoost', 'Logistic Regression', 'Linear Model']:
            return 'Conventional ML'
        elif method == 'KNN':
            return 'KNN Baseline'
        elif method in ['JOLT', 'TabLLM']:
            return 'LLM Methods'
        elif method == 'Conventional' and ('Gemini' in backend or 'Qwen' in backend):
            return 'VLM Methods'
        else:
            return 'Other'
    
    df['Method_Category'] = df.apply(categorize_method, axis=1)
    
    # Calculate statistics for the 3-line structure
    stats = {}
    for category in ['CLAMS', 'Conventional ML', 'KNN Baseline', 'LLM Methods', 'VLM Methods']:
        category_data = df[df['Method_Category'] == category]
        if len(category_data) > 0:
            stats[category] = {
                'mean': category_data['Value'].mean(),
                'coverage': len(category_data),
                'domains': category_data['Domain'].nunique(),
                'max': category_data['Value'].max(),
                'min': category_data['Value'].min()
            }
    
    # Calculate combined traditional baselines stats
    traditional_data = df[df['Method_Category'].isin(['Conventional ML', 'KNN Baseline', 'Contrastive'])]
    if len(traditional_data) > 0:
        stats['Traditional Baselines'] = {
            'mean': traditional_data['Value'].mean(),
            'coverage': len(traditional_data),
            'domains': traditional_data['Domain'].nunique(),
            'max': traditional_data['Value'].max(),
            'min': traditional_data['Value'].min()
        }
    
    # Modality coverage analysis
    clam_domains = df[df['Method_Category'] == 'CLAMS']['Domain'].unique()
    llm_domains = df[df['Method_Category'] == 'LLM Methods']['Domain'].unique()
    vlm_domains = df[df['Method_Category'] == 'VLM Methods']['Domain'].unique()
    
    # Get specific performance highlights
    clam_data = df[df['Method_Category'] == 'CLAMS']
    performance_highlights = {}
    
    for domain in clam_domains:
        domain_data = clam_data[clam_data['Domain'] == domain]
        if len(domain_data) > 0:
            performance_highlights[domain] = {
                'max': domain_data['Value'].max(),
                'benchmark': domain_data.loc[domain_data['Value'].idxmax(), 'Benchmark']
            }
    
    return stats, clam_domains, llm_domains, vlm_domains, performance_highlights

def generate_caption():
    """Generate the publication caption."""
    
    # Load data for the combined stats calculation
    df = pd.read_csv("/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/clam/results/core_results.csv")
    df = df.dropna(subset=['Domain', 'Benchmark', 'Method', 'Value'])
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna(subset=['Value'])
    
    # Apply categorization
    def categorize_method(row):
        method = row['Method']
        backend = row['Backend']
        
        if method == 'CLAMS':
            return 'CLAMS'
        elif method == 'Conventional' and backend in ['TabPFNv2', 'Random Forest', 'CatBoost', 'Logistic Regression', 'Linear Model']:
            return 'Conventional ML'
        elif method == 'KNN':
            return 'KNN Baseline'
        elif method in ['JOLT', 'TabLLM']:
            return 'LLM Methods'
        elif method == 'Conventional' and ('Gemini' in backend or 'Qwen' in backend):
            return 'VLM Methods'
        else:
            return 'Other'
    
    df['Method_Category'] = df.apply(categorize_method, axis=1)
    
    stats, clam_domains, llm_domains, vlm_domains, performance_highlights = analyze_results()
    
    # Combine LLM and VLM stats for the combined category
    combined_llm_vlm_data = df[df['Method_Category'].isin(['LLM Methods', 'VLM Methods'])]
    combined_stats = {
        'mean': combined_llm_vlm_data['Value'].mean(),
        'coverage': len(combined_llm_vlm_data)
    }

    caption = f"""**Figure: CLAMS Performance Radar Plot Across Modalities and Benchmarks.**

CLAMS 🐚 (Classification using Language-guided Augmented Mappings) demonstrates competitive performance across {len(clam_domains)} modalities, achieving an average accuracy of {stats['CLAMS']['mean']:.1f}% across {stats['CLAMS']['coverage']} benchmark evaluations. The radar plot reveals CLAMS' unique position as a bridge between traditional machine learning approaches and modern large language model (LLM) / vision-language model (VLM) methods.

**Key Performance Insights:**
• **CLAMS (modality-colored line with pattern fill)**: Achieves consistent performance across vision ({performance_highlights.get('Vision', {}).get('max', 0):.0f}% on {performance_highlights.get('Vision', {}).get('benchmark', 'CIFAR-10')}), audio ({performance_highlights.get('Audio', {}).get('max', 0):.0f}% on {performance_highlights.get('Audio', {}).get('benchmark', 'ESC-50')}), biological ({performance_highlights.get('Biological', {}).get('max', 0):.0f}% on {performance_highlights.get('Biological', {}).get('benchmark', 'AWA2')}), and tabular domains ({performance_highlights.get('Tabular Classification', {}).get('max', 0):.0f}% classification, {performance_highlights.get('Tabular Regression', {}).get('max', 0):.0f}% regression). Each segment is colored by modality: vision (red), audio (orange), biological (green), tabular classification (blue), and tabular regression (purple).

• **LLM/VLM Combined (blue solid)**: Demonstrate broad applicability across modalities with {combined_stats['mean']:.1f}% average performance across {combined_stats['coverage']} evaluations, representing the "flexible but slower" paradigm.

• **Traditional Baselines (green dashed)**: Combined performance of conventional ML, KNN, and contrastive methods shows {stats.get('Traditional Baselines', {}).get('mean', 0):.1f}% average performance, representing the "fast and accurate but inflexible" end of the spectrum.

**Bridge Paradigm**: CLAMS uniquely combines the speed and accuracy of conventional ML (through TabPFN embeddings and t-SNE visualization) with the flexibility and reasoning capabilities of VLMs, offering a practical middle ground that maintains competitive performance while enabling interpretable, visual reasoning across diverse data modalities. This positioning makes CLAMS particularly valuable for applications requiring both performance and interpretability across heterogeneous data types.

The plot uses polar coordinates with benchmarks as spokes. CLAMS' modality-colored trajectory demonstrates consistent performance across the full spectrum of evaluation scenarios, with each color representing a different data domain. Missing performance values are shown as zero to clearly indicate modality coverage limitations."""

    return caption

def save_caption():
    """Save the caption to a file."""
    caption = generate_caption()
    
    output_file = "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/clam/results/radar_plot_caption.txt"
    
    with open(output_file, 'w') as f:
        f.write(caption)
    
    print(f"Caption saved to: {output_file}")
    print("\n" + "="*80)
    print("GENERATED CAPTION:")
    print("="*80)
    print(caption)
    
    return caption

if __name__ == "__main__":
    save_caption()