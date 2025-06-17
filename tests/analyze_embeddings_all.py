#!/usr/bin/env python3
"""
Enhanced script to visualize embeddings from ALL real and synthetic datasets.
Shows PCA/t-SNE for each dataset to compare structure across all examples.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_embeddings_from_directory(directory_path):
    """Load all .npz embedding files from a directory."""
    embeddings_data = {}
    
    for file_path in Path(directory_path).glob("*.npz"):
        print(f"Loading {file_path.name}...")
        try:
            data = np.load(file_path, allow_pickle=True)
            embeddings_data[file_path.stem] = {
                'data': data,
                'file_path': str(file_path)
            }
        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")
    
    return embeddings_data

def create_range_comparison_plot(real_embeddings, syn_embeddings):
    """Create a plot comparing the min/max ranges of all datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Collect ranges for real data
    real_ranges = []
    real_names = []
    for name, emb_info in real_embeddings.items():
        data = emb_info['data']
        if 'train_embeddings' in data:
            emb = data['train_embeddings']
            real_ranges.append([np.min(emb), np.max(emb)])
            real_names.append(name.split('_')[0])  # Just the dataset ID
    
    # Collect ranges for synthetic data
    syn_ranges = []
    syn_names = []
    for name, emb_info in syn_embeddings.items():
        data = emb_info['data']
        if 'train_embeddings' in data:
            emb = data['train_embeddings']
            syn_ranges.append([np.min(emb), np.max(emb)])
            syn_names.append(name.split('_')[1])  # Just the episode number
    
    # Plot real data ranges
    real_ranges = np.array(real_ranges)
    positions = np.arange(len(real_names))
    ax1.barh(positions, real_ranges[:, 1] - real_ranges[:, 0], 
             left=real_ranges[:, 0], height=0.6, color='coral', alpha=0.8)
    ax1.set_yticks(positions)
    ax1.set_yticklabels(real_names)
    ax1.set_xlabel('Embedding Value Range')
    ax1.set_title('Real Data: Min-Max Ranges')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot synthetic data ranges
    syn_ranges = np.array(syn_ranges)
    positions = np.arange(len(syn_names))
    ax2.barh(positions, syn_ranges[:, 1] - syn_ranges[:, 0], 
             left=syn_ranges[:, 0], height=0.6, color='skyblue', alpha=0.8)
    ax2.set_yticks(positions)
    ax2.set_yticklabels(syn_names)
    ax2.set_xlabel('Embedding Value Range')
    ax2.set_title('Synthetic Data: Min-Max Ranges')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Make x-axes same scale for comparison
    all_min = min(np.min(real_ranges), np.min(syn_ranges))
    all_max = max(np.max(real_ranges), np.max(syn_ranges))
    ax1.set_xlim(all_min - 1, all_max + 1)
    ax2.set_xlim(all_min - 1, all_max + 1)
    
    plt.tight_layout()
    plt.savefig('embedding_ranges_comparison.png', dpi=150)
    print("Saved range comparison to 'embedding_ranges_comparison.png'")

def visualize_all_embeddings_structure(real_embeddings, syn_embeddings, max_samples=500):
    """Create PCA/t-SNE visualizations for ALL datasets."""
    
    # Count datasets
    n_real = len(real_embeddings)
    n_syn = len(syn_embeddings)
    
    # Create figure with subplots for all datasets
    fig_pca = plt.figure(figsize=(20, max(12, 3 * max(n_real, n_syn))))
    fig_tsne = plt.figure(figsize=(20, max(12, 3 * max(n_real, n_syn))))
    
    # Process real datasets
    print("\nProcessing PCA/t-SNE for real datasets...")
    for idx, (name, emb_info) in enumerate(sorted(real_embeddings.items())):
        data = emb_info['data']
        if 'train_embeddings' not in data:
            continue
            
        emb = data['train_embeddings']
        labels = data['y_train_sample'] if 'y_train_sample' in data else None
        
        # Sample data if too large
        n_samples = min(max_samples, len(emb))
        indices = np.random.choice(len(emb), n_samples, replace=False)
        emb_sample = emb[indices]
        labels_sample = labels[indices] if labels is not None else None
        
        # Get dataset info
        dataset_id = name.split('_')[0]
        n_classes = len(np.unique(labels)) if labels is not None else 0
        metadata = data['metadata'].item() if 'metadata' in data else {}
        n_features = metadata.get('n_features', 'N/A')
        
        # PCA
        ax_pca = fig_pca.add_subplot(max(n_real, n_syn), 2, idx * 2 + 1)
        pca = PCA(n_components=2)
        emb_pca = pca.fit_transform(emb_sample)
        
        scatter = ax_pca.scatter(emb_pca[:, 0], emb_pca[:, 1], 
                                c=labels_sample, cmap='tab10', alpha=0.6, s=20)
        ax_pca.set_title(f'Real {dataset_id}: {n_features} feat, {n_classes} class\n'
                        f'PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}')
        ax_pca.set_xlabel('PC1')
        ax_pca.set_ylabel('PC2')
        
        # t-SNE
        ax_tsne = fig_tsne.add_subplot(max(n_real, n_syn), 2, idx * 2 + 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
        emb_tsne = tsne.fit_transform(emb_sample)
        
        scatter = ax_tsne.scatter(emb_tsne[:, 0], emb_tsne[:, 1], 
                                 c=labels_sample, cmap='tab10', alpha=0.6, s=20)
        ax_tsne.set_title(f'Real {dataset_id}: {n_features} feat, {n_classes} class')
        ax_tsne.set_xlabel('t-SNE 1')
        ax_tsne.set_ylabel('t-SNE 2')
    
    # Process synthetic datasets
    print("\nProcessing PCA/t-SNE for synthetic datasets...")
    for idx, (name, emb_info) in enumerate(sorted(syn_embeddings.items())):
        data = emb_info['data']
        if 'train_embeddings' not in data:
            continue
            
        emb = data['train_embeddings']
        labels = data['y_train_sample'] if 'y_train_sample' in data else None
        
        # Sample data if too large
        n_samples = min(max_samples, len(emb))
        indices = np.random.choice(len(emb), n_samples, replace=False)
        emb_sample = emb[indices]
        labels_sample = labels[indices] if labels is not None else None
        
        # Get dataset info
        episode = name.split('_')[1]
        n_classes = len(np.unique(labels)) if labels is not None else 0
        metadata = data['metadata'].item() if 'metadata' in data else {}
        n_features = metadata.get('n_features', 'N/A')
        
        # PCA
        ax_pca = fig_pca.add_subplot(max(n_real, n_syn), 2, idx * 2 + 2)
        pca = PCA(n_components=2)
        emb_pca = pca.fit_transform(emb_sample)
        
        scatter = ax_pca.scatter(emb_pca[:, 0], emb_pca[:, 1], 
                                c=labels_sample, cmap='tab10', alpha=0.6, s=20)
        ax_pca.set_title(f'Syn {episode}: {n_features} feat, {n_classes} class\n'
                        f'PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}')
        ax_pca.set_xlabel('PC1')
        ax_pca.set_ylabel('PC2')
        
        # t-SNE
        ax_tsne = fig_tsne.add_subplot(max(n_real, n_syn), 2, idx * 2 + 2)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
        emb_tsne = tsne.fit_transform(emb_sample)
        
        scatter = ax_tsne.scatter(emb_tsne[:, 0], emb_tsne[:, 1], 
                                 c=labels_sample, cmap='tab10', alpha=0.6, s=20)
        ax_tsne.set_title(f'Syn {episode}: {n_features} feat, {n_classes} class')
        ax_tsne.set_xlabel('t-SNE 1')
        ax_tsne.set_ylabel('t-SNE 2')
    
    # Add main titles and save
    fig_pca.suptitle('PCA Projections: Real (left) vs Synthetic (right)', fontsize=16, y=0.995)
    fig_tsne.suptitle('t-SNE Projections: Real (left) vs Synthetic (right)', fontsize=16, y=0.995)
    
    fig_pca.tight_layout()
    fig_tsne.tight_layout()
    
    fig_pca.savefig('all_embeddings_pca.png', dpi=150, bbox_inches='tight')
    fig_tsne.savefig('all_embeddings_tsne.png', dpi=150, bbox_inches='tight')
    
    print("Saved PCA visualization to 'all_embeddings_pca.png'")
    print("Saved t-SNE visualization to 'all_embeddings_tsne.png'")

def create_statistics_heatmap(real_embeddings, syn_embeddings):
    """Create a heatmap comparing key statistics across all datasets."""
    import pandas as pd
    
    stats_data = []
    
    # Collect stats for real datasets
    for name, emb_info in sorted(real_embeddings.items()):
        data = emb_info['data']
        if 'train_embeddings' in data:
            emb = data['train_embeddings']
            labels = data['y_train_sample'] if 'y_train_sample' in data else None
            metadata = data['metadata'].item() if 'metadata' in data else {}
            
            stats_data.append({
                'Dataset': f"Real_{name.split('_')[0]}",
                'Type': 'Real',
                'Min': np.min(emb),
                'Max': np.max(emb),
                'Range': np.max(emb) - np.min(emb),
                'Mean': np.mean(emb),
                'Std': np.std(emb),
                'Features': metadata.get('n_features', 0),
                'Classes': len(np.unique(labels)) if labels is not None else 0
            })
    
    # Collect stats for synthetic datasets
    for name, emb_info in sorted(syn_embeddings.items()):
        data = emb_info['data']
        if 'train_embeddings' in data:
            emb = data['train_embeddings']
            labels = data['y_train_sample'] if 'y_train_sample' in data else None
            metadata = data['metadata'].item() if 'metadata' in data else {}
            
            stats_data.append({
                'Dataset': f"Syn_{name.split('_')[1]}",
                'Type': 'Synthetic',
                'Min': np.min(emb),
                'Max': np.max(emb),
                'Range': np.max(emb) - np.min(emb),
                'Mean': np.mean(emb),
                'Std': np.std(emb),
                'Features': metadata.get('n_features', 0),
                'Classes': len(np.unique(labels)) if labels is not None else 0
            })
    
    # Create dataframe and pivot for heatmap
    df = pd.DataFrame(stats_data)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Select numeric columns for heatmap
    numeric_cols = ['Min', 'Max', 'Range', 'Mean', 'Std']
    heatmap_data = df.set_index('Dataset')[numeric_cols].T
    
    # Normalize each row to [0, 1] for better visualization
    heatmap_normalized = heatmap_data.div(heatmap_data.max(axis=1), axis=0)
    
    sns.heatmap(heatmap_normalized, annot=heatmap_data, fmt='.2f', 
                cmap='coolwarm', center=0.5, cbar_kws={'label': 'Normalized Value'})
    
    ax.set_title('Embedding Statistics Comparison: Real vs Synthetic Datasets')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Statistic')
    
    # Add vertical line to separate real from synthetic
    n_real = len([d for d in stats_data if d['Type'] == 'Real'])
    ax.axvline(x=n_real, color='black', linewidth=2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('embedding_statistics_heatmap.png', dpi=150)
    print("Saved statistics heatmap to 'embedding_statistics_heatmap.png'")
    
    # Save detailed statistics
    df.to_csv('all_datasets_statistics.csv', index=False)
    print("Saved detailed statistics to 'all_datasets_statistics.csv'")

def main():
    # Define paths
    real_path = "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/llata/embeddings/realdata"
    syn_path = "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/llata/embeddings/syndata"
    
    print("Loading embeddings...")
    
    # Load embeddings
    real_embeddings = load_embeddings_from_directory(real_path)
    syn_embeddings = load_embeddings_from_directory(syn_path)
    
    print(f"\nLoaded {len(real_embeddings)} real datasets")
    print(f"Loaded {len(syn_embeddings)} synthetic datasets")
    
    # Create range comparison
    create_range_comparison_plot(real_embeddings, syn_embeddings)
    
    # Visualize all embeddings
    visualize_all_embeddings_structure(real_embeddings, syn_embeddings)
    
    # Create statistics heatmap
    create_statistics_heatmap(real_embeddings, syn_embeddings)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()