#!/usr/bin/env python3
"""
Script to analyze and compare embeddings from real and synthetic datasets.
This helps understand the differences between embeddings that the VQ model
successfully learns from (synthetic) vs those it fails on (real).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd
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

def analyze_embedding_statistics(embeddings_dict, dataset_type):
    """Compute statistics for embeddings."""
    print(f"\n{'='*60}")
    print(f"Statistics for {dataset_type} embeddings")
    print(f"{'='*60}")
    
    all_stats = []
    
    for name, emb_info in embeddings_dict.items():
        data = emb_info['data']
        
        # Extract metadata if available
        metadata = data['metadata'].item() if 'metadata' in data else {}
        
        # Analyze each embedding type
        for emb_type in ['train_embeddings', 'val_embeddings', 'test_embeddings']:
            if emb_type in data:
                embeddings = data[emb_type]
                
                stats = {
                    'dataset': name,
                    'type': emb_type,
                    'shape': embeddings.shape,
                    'mean': np.mean(embeddings),
                    'std': np.std(embeddings),
                    'min': np.min(embeddings),
                    'max': np.max(embeddings),
                    'median': np.median(embeddings),
                    'sparsity': np.mean(np.abs(embeddings) < 1e-6),  # Fraction near zero
                    'n_features': metadata.get('n_features', 'N/A'),
                    'n_classes': len(np.unique(data['y_train_sample'])) if 'y_train_sample' in data else 'N/A'
                }
                all_stats.append(stats)
                
                print(f"\n{name} - {emb_type}:")
                print(f"  Shape: {stats['shape']}")
                print(f"  Mean: {stats['mean']:.6f}")
                print(f"  Std: {stats['std']:.6f}")
                print(f"  Min/Max: [{stats['min']:.6f}, {stats['max']:.6f}]")
                print(f"  Sparsity (% near zero): {stats['sparsity']*100:.2f}%")
                print(f"  Original features: {stats['n_features']}")
                print(f"  Number of classes: {stats['n_classes']}")
    
    return pd.DataFrame(all_stats)

def compare_distributions(real_embeddings, syn_embeddings):
    """Compare the distributions of real vs synthetic embeddings."""
    print(f"\n{'='*60}")
    print("Comparing Real vs Synthetic Embedding Distributions")
    print(f"{'='*60}")
    
    # Collect all embeddings
    real_all = []
    syn_all = []
    
    for name, emb_info in real_embeddings.items():
        data = emb_info['data']
        if 'train_embeddings' in data:
            real_all.append(data['train_embeddings'].flatten())
    
    for name, emb_info in syn_embeddings.items():
        data = emb_info['data']
        if 'train_embeddings' in data:
            syn_all.append(data['train_embeddings'].flatten())
    
    if real_all and syn_all:
        real_all = np.concatenate(real_all)
        syn_all = np.concatenate(syn_all)
        
        # Sample for visualization (too many points otherwise)
        sample_size = min(10000, len(real_all), len(syn_all))
        real_sample = np.random.choice(real_all, sample_size, replace=False)
        syn_sample = np.random.choice(syn_all, sample_size, replace=False)
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histograms
        axes[0, 0].hist(real_sample, bins=50, alpha=0.6, label='Real', density=True)
        axes[0, 0].hist(syn_sample, bins=50, alpha=0.6, label='Synthetic', density=True)
        axes[0, 0].set_xlabel('Embedding Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Embedding Value Distributions')
        axes[0, 0].legend()
        
        # Box plots
        axes[0, 1].boxplot([real_sample, syn_sample], labels=['Real', 'Synthetic'])
        axes[0, 1].set_ylabel('Embedding Value')
        axes[0, 1].set_title('Box Plot Comparison')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(real_sample, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot: Real Embeddings')
        
        stats.probplot(syn_sample, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot: Synthetic Embeddings')
        
        plt.tight_layout()
        plt.savefig('embedding_distributions.png', dpi=150)
        print("\nSaved distribution comparison to 'embedding_distributions.png'")
        
        # Statistical tests
        print("\nStatistical Tests:")
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(real_sample, syn_sample)
        print(f"  Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={ks_pval:.4e}")
        
        # Mann-Whitney U test
        mw_stat, mw_pval = stats.mannwhitneyu(real_sample, syn_sample)
        print(f"  Mann-Whitney U test: statistic={mw_stat:.4f}, p-value={mw_pval:.4e}")

def analyze_embedding_structure(real_embeddings, syn_embeddings):
    """Analyze the structure of embeddings using dimensionality reduction."""
    print(f"\n{'='*60}")
    print("Analyzing Embedding Structure with PCA")
    print(f"{'='*60}")
    
    # Collect first dataset from each type for detailed analysis
    real_example = None
    syn_example = None
    
    for name, emb_info in real_embeddings.items():
        if 'train_embeddings' in emb_info['data']:
            real_example = (name, emb_info['data'])
            break
    
    for name, emb_info in syn_embeddings.items():
        if 'train_embeddings' in emb_info['data']:
            syn_example = (name, emb_info['data'])
            break
    
    if real_example and syn_example:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for idx, (name, data, ax_row) in enumerate([
            (real_example[0], real_example[1], axes[0]),
            (syn_example[0], syn_example[1], axes[1])
        ]):
            emb = data['train_embeddings']
            labels = data['y_train_sample'] if 'y_train_sample' in data else None
            
            # Limit samples for visualization
            n_samples = min(1000, len(emb))
            indices = np.random.choice(len(emb), n_samples, replace=False)
            emb_sample = emb[indices]
            labels_sample = labels[indices] if labels is not None else None
            
            # PCA
            pca = PCA(n_components=2)
            emb_pca = pca.fit_transform(emb_sample)
            
            # TSNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            emb_tsne = tsne.fit_transform(emb_sample)
            
            # Plot PCA
            scatter = ax_row[0].scatter(emb_pca[:, 0], emb_pca[:, 1], 
                                       c=labels_sample, cmap='tab10', alpha=0.6, s=20)
            ax_row[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax_row[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax_row[0].set_title(f'PCA - {"Real" if idx == 0 else "Synthetic"}: {name}')
            if labels_sample is not None:
                ax_row[0].legend(*scatter.legend_elements(), title="Classes", loc='best')
            
            # Plot t-SNE
            scatter = ax_row[1].scatter(emb_tsne[:, 0], emb_tsne[:, 1], 
                                       c=labels_sample, cmap='tab10', alpha=0.6, s=20)
            ax_row[1].set_xlabel('t-SNE 1')
            ax_row[1].set_ylabel('t-SNE 2')
            ax_row[1].set_title(f't-SNE - {"Real" if idx == 0 else "Synthetic"}: {name}')
            
            # Print explained variance
            total_var = np.sum(pca.explained_variance_ratio_[:min(10, len(pca.explained_variance_ratio_))])
            print(f"\n{name} ({'Real' if idx == 0 else 'Synthetic'}):")
            print(f"  Variance explained by first 10 PCs: {total_var:.2%}")
            print(f"  Variance explained by PC1: {pca.explained_variance_ratio_[0]:.2%}")
            print(f"  Variance explained by PC2: {pca.explained_variance_ratio_[1]:.2%}")
        
        plt.tight_layout()
        plt.savefig('embedding_structure.png', dpi=150)
        print("\nSaved structure analysis to 'embedding_structure.png'")

def analyze_class_separability(embeddings_dict, dataset_type, max_datasets=3):
    """Analyze how well classes are separated in the embedding space."""
    print(f"\n{'='*60}")
    print(f"Class Separability Analysis for {dataset_type}")
    print(f"{'='*60}")
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import silhouette_score
    
    results = []
    
    for i, (name, emb_info) in enumerate(embeddings_dict.items()):
        if i >= max_datasets:
            break
            
        data = emb_info['data']
        if 'train_embeddings' in data and 'y_train_sample' in data:
            emb = data['train_embeddings']
            labels = data['y_train_sample']
            
            # Compute silhouette score
            if len(np.unique(labels)) > 1:
                try:
                    # Sample if too large
                    if len(emb) > 1000:
                        indices = np.random.choice(len(emb), 1000, replace=False)
                        emb_sample = emb[indices]
                        labels_sample = labels[indices]
                    else:
                        emb_sample = emb
                        labels_sample = labels
                    
                    silhouette = silhouette_score(emb_sample, labels_sample)
                    
                    # Try LDA
                    n_classes = len(np.unique(labels))
                    lda = LinearDiscriminantAnalysis()
                    lda.fit(emb_sample, labels_sample)
                    lda_score = lda.score(emb_sample, labels_sample)
                    
                    results.append({
                        'dataset': name,
                        'n_classes': n_classes,
                        'silhouette_score': silhouette,
                        'lda_accuracy': lda_score
                    })
                    
                    print(f"\n{name}:")
                    print(f"  Number of classes: {n_classes}")
                    print(f"  Silhouette score: {silhouette:.4f}")
                    print(f"  LDA accuracy: {lda_score:.4f}")
                    
                except Exception as e:
                    print(f"\n{name}: Error computing separability - {e}")
    
    return pd.DataFrame(results)

def main():
    # Define paths
    real_path = "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/clam/embeddings/realdata"
    syn_path = "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/clam/embeddings/syndata"
    
    print("Loading embeddings...")
    
    # Load embeddings
    real_embeddings = load_embeddings_from_directory(real_path)
    syn_embeddings = load_embeddings_from_directory(syn_path)
    
    print(f"\nLoaded {len(real_embeddings)} real datasets")
    print(f"Loaded {len(syn_embeddings)} synthetic datasets")
    
    # Analyze statistics
    real_stats = analyze_embedding_statistics(real_embeddings, "Real Data")
    syn_stats = analyze_embedding_statistics(syn_embeddings, "Synthetic Data")
    
    # Compare distributions
    compare_distributions(real_embeddings, syn_embeddings)
    
    # Analyze structure
    analyze_embedding_structure(real_embeddings, syn_embeddings)
    
    # Analyze class separability
    real_separability = analyze_class_separability(real_embeddings, "Real Data")
    syn_separability = analyze_class_separability(syn_embeddings, "Synthetic Data")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    
    if len(real_stats) > 0 and len(syn_stats) > 0:
        print("\nAverage Statistics:")
        print(f"Real Data:")
        print(f"  Mean embedding value: {real_stats['mean'].mean():.6f} ± {real_stats['mean'].std():.6f}")
        print(f"  Mean std dev: {real_stats['std'].mean():.6f} ± {real_stats['std'].std():.6f}")
        print(f"  Mean sparsity: {real_stats['sparsity'].mean()*100:.2f}%")
        
        print(f"\nSynthetic Data:")
        print(f"  Mean embedding value: {syn_stats['mean'].mean():.6f} ± {syn_stats['mean'].std():.6f}")
        print(f"  Mean std dev: {syn_stats['std'].mean():.6f} ± {syn_stats['std'].std():.6f}")
        print(f"  Mean sparsity: {syn_stats['sparsity'].mean()*100:.2f}%")
    
    if len(real_separability) > 0 and len(syn_separability) > 0:
        print("\nAverage Class Separability:")
        print(f"Real Data:")
        print(f"  Mean silhouette score: {real_separability['silhouette_score'].mean():.4f}")
        print(f"  Mean LDA accuracy: {real_separability['lda_accuracy'].mean():.4f}")
        
        print(f"\nSynthetic Data:")
        print(f"  Mean silhouette score: {syn_separability['silhouette_score'].mean():.4f}")
        print(f"  Mean LDA accuracy: {syn_separability['lda_accuracy'].mean():.4f}")
    
    # Save detailed results
    real_stats.to_csv('real_embeddings_stats.csv', index=False)
    syn_stats.to_csv('synthetic_embeddings_stats.csv', index=False)
    print("\n\nDetailed statistics saved to 'real_embeddings_stats.csv' and 'synthetic_embeddings_stats.csv'")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()