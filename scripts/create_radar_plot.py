#!/usr/bin/env python3
"""
Create a comprehensive radar plot from core_results.csv showing CLAM's performance
across modalities compared to traditional ML and LLM/VLM approaches.

The radar plot visualizes CLAM as a bridge between fast conventional ML and 
flexible but slower LLM/VLM approaches across multiple modalities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
RESULTS_FILE = "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/clam/results/core_results.csv"
OUTPUT_DIR = "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/clam/results"

def load_and_preprocess_data():
    """Load and preprocess the core results data."""
    df = pd.read_csv(RESULTS_FILE)
    
    # Remove empty rows
    df = df.dropna(subset=['Domain', 'Benchmark', 'Method', 'Value'])
    
    # Handle missing values and convert to numeric
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna(subset=['Value'])
    
    # Create a combined benchmark identifier
    df['Benchmark_Full'] = df['Domain'] + ': ' + df['Benchmark']
    
    # Group methods into categories for visualization
    def categorize_method(row):
        method = row['Method']
        backend = row['Backend']
        
        if method == 'CLAMS':
            return 'CLAM'
        elif method == 'Conventional' and backend in ['TabPFNv2', 'Random Forest', 'CatBoost', 'Logistic Regression', 'Linear Model']:
            return 'Conventional ML'
        elif method == 'KNN':
            return 'KNN Baseline'
        elif method == 'Contrastive':
            return 'Contrastive'
        elif method in ['JOLT', 'TabLLM']:
            return 'LLM/Serialization'
        elif method == 'Conventional' and 'Gemini' in backend:
            return 'VLM/Gemini'
        elif method == 'Conventional' and 'Qwen' in backend:
            return 'VLM/Qwen'
        else:
            return 'Other'
    
    df['Method_Category'] = df.apply(categorize_method, axis=1)
    
    return df

def aggregate_results(df):
    """Aggregate results by method category and benchmark."""
    # Group similar methods together - now with only 3 total lines
    aggregated_groups = {
        'CLAMS': ['CLAM'],
        'LLM/VLM Combined': ['VLM/Gemini', 'VLM/Qwen', 'LLM/Serialization'], 
        'Traditional Baselines': ['Conventional ML', 'KNN Baseline', 'Contrastive']
    }
    
    results = {}
    
    for group_name, method_cats in aggregated_groups.items():
        group_data = df[df['Method_Category'].isin(method_cats)]
        
        if len(group_data) == 0:
            continue
            
        # Average across method categories within the group
        group_agg = group_data.groupby('Benchmark_Full')['Value'].mean().reset_index()
        group_agg['Method_Group'] = group_name
        
        results[group_name] = group_agg
    
    return results

def create_radar_data(aggregated_results):
    """Create radar plot data matrix with zeros for missing values."""
    # Get all unique benchmarks
    all_benchmarks = set()
    for group_data in aggregated_results.values():
        all_benchmarks.update(group_data['Benchmark_Full'].tolist())
    
    benchmarks = sorted(list(all_benchmarks))
    
    # Create data matrix - use 0 instead of NaN for missing values
    radar_data = {}
    
    for group_name, group_data in aggregated_results.items():
        values = []
        for benchmark in benchmarks:
            benchmark_data = group_data[group_data['Benchmark_Full'] == benchmark]
            if len(benchmark_data) > 0:
                values.append(benchmark_data['Value'].iloc[0])
            else:
                values.append(0)  # Use 0 instead of NaN
        radar_data[group_name] = values
    
    return pd.DataFrame(radar_data, index=benchmarks)

def setup_radar_plot(num_vars):
    """Set up the radar plot structure."""
    # Calculate angles for each spoke
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Create figure and polar subplot
    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(projection='polar'))
    
    return fig, ax, angles

def plot_radar_rings(ax, radar_df, angles):
    """Plot the radar chart with multiple rings."""
    
    # Define colors and styles for each method group (only 3 now)
    plot_configs = {
        'CLAMS': {
            'linewidth': 5, 
            'alpha': 0.9, 
            'linestyle': '-',
            'marker': 'o',
            'markersize': 14,
            'fill_alpha': 0.15,
            'use_rainbow': True,  # Special flag for rainbow coloring
            'use_pattern_fill': True  # Use pattern instead of solid fill
        },
        'LLM/VLM Combined': {
            'color': 'black', 
            'linewidth': 3, 
            'alpha': 0.8, 
            'linestyle': ':',  # Dotted line
            'marker': 's',
            'markersize': 10,
            'fill_alpha': 0.1
        },
        'Traditional Baselines': {
            'color': 'black', 
            'linewidth': 3, 
            'alpha': 0.7, 
            'linestyle': '--',  # Dashed line
            'marker': 'D',
            'markersize': 9,
            'fill_alpha': 0.08
        }
    }
    
    # Plot order (CLAMS last to appear on top)
    plot_order = ['Traditional Baselines', 'LLM/VLM Combined', 'CLAMS']
    
    plotted_methods = []
    
    for method in plot_order:
        if method not in radar_df.columns:
            continue
            
        values = radar_df[method].tolist()
        
        # All values are now numbers (including 0s), so process all of them
        plot_values = values + [values[0]]  # Close the polygon
        plot_angles = angles  # angles already includes the closing angle
        
        config = plot_configs[method]
        
        # Special handling for CLAMS rainbow coloring
        if config.get('use_rainbow', False):
            # Create modality-based colors for CLAMS
            import matplotlib.colors as mcolors
            import matplotlib.cm as cm
            
            # Define modality colors based on benchmark names
            benchmarks = radar_df.index.tolist()
            modality_colors = {
                'Vision': '#E74C3C',    # Red
                'Audio': '#F39C12',     # Orange  
                'Biological': '#27AE60', # Green
                'Tabular Classification': '#3498DB', # Blue
                'Tabular Regression': '#9B59B6'  # Purple
            }
            
            num_segments = len(plot_values) - 1
            
            # Plot each segment with modality-specific color
            for i in range(num_segments):
                benchmark = benchmarks[i] if i < len(benchmarks) else benchmarks[0]
                modality = benchmark.split(':')[0]
                segment_color = modality_colors.get(modality, '#E74C3C')
                
                ax.plot([plot_angles[i], plot_angles[i+1]], 
                       [plot_values[i], plot_values[i+1]],
                       color=segment_color,
                       linewidth=config['linewidth'],
                       alpha=config['alpha'],
                       linestyle=config['linestyle'])
            
            # Plot markers with modality colors
            for i in range(num_segments):
                benchmark = benchmarks[i] if i < len(benchmarks) else benchmarks[0]
                modality = benchmark.split(':')[0]
                segment_color = modality_colors.get(modality, '#E74C3C')
                
                ax.plot(plot_angles[i], plot_values[i],
                       marker=config['marker'],
                       markersize=config['markersize'],
                       color=segment_color,
                       markerfacecolor=segment_color,
                       markeredgecolor='white',
                       markeredgewidth=2)
            
            # Fill with pattern instead of solid color
            if config.get('use_pattern_fill', False):
                ax.fill(plot_angles, plot_values, 
                       color='lightgray', 
                       alpha=config['fill_alpha'],
                       hatch='///')  # Diagonal lines pattern
            else:
                ax.fill(plot_angles, plot_values, 
                       color='red', 
                       alpha=config['fill_alpha'])
            
            # Add to legend with red color for consistency
            ax.plot([], [], color='red', linewidth=config['linewidth'], 
                   linestyle=config['linestyle'], marker=config['marker'],
                   markersize=config['markersize'], label=method)
                   
        else:
            # Standard single-color plotting
            line = ax.plot(plot_angles, plot_values, 
                          color=config['color'],
                          linewidth=config['linewidth'],
                          alpha=config['alpha'],
                          linestyle=config['linestyle'],
                          marker=config['marker'],
                          markersize=config['markersize'],
                          label=method,
                          markerfacecolor=config['color'],
                          markeredgecolor='white',
                          markeredgewidth=1.5)
            
            # Fill the area for main methods
            if method in ['CLAMS', 'LLM/VLM Combined'] and len(plot_values) > 2:
                ax.fill(plot_angles, plot_values, 
                       color=config['color'], 
                       alpha=config['fill_alpha'])
        
        plotted_methods.append(method)
    
    return plotted_methods

def style_radar_plot(ax, angles, benchmarks, radar_df):
    """Apply styling to the radar plot."""
    
    # Set the angle of the first spoke
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set spoke labels
    ax.set_xticks(angles[:-1])
    
    # Create clean benchmark labels with CLAMS scores
    clean_labels = []
    clams_values = radar_df['CLAMS'].tolist() if 'CLAMS' in radar_df.columns else [0] * len(benchmarks)
    
    for i, benchmark in enumerate(benchmarks):
        clams_score = clams_values[i] if i < len(clams_values) else 0
        
        # Shorten long labels
        if len(benchmark) > 25:
            parts = benchmark.split(': ')
            if len(parts) == 2:
                domain = parts[0]
                name = parts[1]
                if len(name) > 15:
                    name = name[:12] + '...'
                label_text = f"{domain}:\n{name}\n({clams_score:.1f}%)"
            else:
                label_text = f"{benchmark[:22]}...\n({clams_score:.1f}%)"
        else:
            benchmark_clean = benchmark.replace(': ', ':\n')
            label_text = f"{benchmark_clean}\n({clams_score:.1f}%)"
            
        clean_labels.append(label_text)
    
    ax.set_xticklabels(clean_labels, fontsize=10, fontweight='bold')
    
    # Style the radial grid
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10, alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    # Style the radial lines
    ax.set_rgrids([20, 40, 60, 80, 100], alpha=0.3)

def add_legend_and_title(fig, ax, plotted_methods):
    """Add legend and title to the plot."""
    
    import matplotlib.patches as mpatches
    
    # Create a custom legend with method descriptions
    legend_labels = {
        'CLAMS': 'CLAMS (Vision-Language Classification)',
        'LLM/VLM Combined': 'LLM/VLM Combined (black dotted)', 
        'Traditional Baselines': 'Traditional Baselines (black dashed)'
    }
    
    # Filter to only plotted methods
    filtered_labels = [legend_labels.get(method, method) for method in plotted_methods if method in legend_labels]
    
    # Create main legend
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
                      fontsize=12, frameon=True, fancybox=True, shadow=True,
                      title='Methods\n(Numbers show CLAM-3B scores)')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.get_title().set_fontweight('bold')
    
    # Add modality color legend for CLAMS
    modality_colors = {
        'Vision': '#E74C3C',    # Red
        'Audio': '#F39C12',     # Orange  
        'Biological': '#27AE60', # Green
        'Tabular Classification': '#3498DB', # Blue
        'Tabular Regression': '#9B59B6'  # Purple
    }
    
    # Create modality legend patches
    modality_patches = [mpatches.Patch(color=color, label=f'{domain} Domain') 
                       for domain, color in modality_colors.items()]
    
    # Add modality legend
    modality_legend = ax.legend(handles=modality_patches, 
                               loc='upper left', bbox_to_anchor=(-0.2, 1.1),
                               fontsize=10, frameon=True, fancybox=True, shadow=True,
                               title='CLAMS Modality Colors')
    modality_legend.get_frame().set_facecolor('white')
    modality_legend.get_frame().set_alpha(0.9)
    
    # Add the main legend back (matplotlib only shows one legend by default)
    ax.add_artist(legend)
    
    # Add main title with shell symbol
    title_text = '🚀 CLAMS: Bridging Conventional ML and Modern LLM/VLM Approaches\nPerformance Across Modalities and Benchmarks'
    plt.suptitle(title_text, fontsize=18, fontweight='bold', y=0.95)

# Removed add_annotations function since scores are now in spoke labels

def create_modality_sections(ax, angles, benchmarks):
    """Add visual sections for different modalities."""
    
    # Group benchmarks by modality
    modality_ranges = {}
    current_modality = None
    start_idx = 0
    
    for i, benchmark in enumerate(benchmarks):
        modality = benchmark.split(':')[0]
        
        if modality != current_modality:
            if current_modality is not None:
                modality_ranges[current_modality] = (start_idx, i-1)
            current_modality = modality
            start_idx = i
    
    # Add the last modality
    if current_modality is not None:
        modality_ranges[current_modality] = (start_idx, len(benchmarks)-1)
    
    # Add colored arcs for each modality
    modality_colors = {
        'Vision': '#3498DB',
        'Audio': '#E74C3C', 
        'Biological': '#27AE60',
        'Tabular Classification': '#9B59B6',
        'Tabular Regression': '#F39C12'
    }
    
    for modality, (start, end) in modality_ranges.items():
        if end >= start and modality in modality_colors:
            start_angle = angles[start]
            end_angle = angles[end + 1] if end + 1 < len(angles) else angles[end]
            
            # Create arc
            arc_angles = np.linspace(start_angle, end_angle, 100)
            ax.fill_between(arc_angles, 105, 110, 
                           color=modality_colors[modality], 
                           alpha=0.6, label=f'{modality} Domain')

def main():
    """Main function to create the radar plot."""
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    print("Aggregating results...")
    aggregated_results = aggregate_results(df)
    
    print("Creating radar data matrix...")
    radar_df = create_radar_data(aggregated_results)
    
    print("Available methods:", radar_df.columns.tolist())
    print("Available benchmarks:", radar_df.index.tolist())
    
    # Create the radar plot
    print("Creating radar plot...")
    benchmarks = radar_df.index.tolist()
    num_vars = len(benchmarks)
    
    fig, ax, angles = setup_radar_plot(num_vars)
    
    # Plot the data
    plotted_methods = plot_radar_rings(ax, radar_df, angles)
    
    # Style the plot
    style_radar_plot(ax, angles, benchmarks, radar_df)
    
    # Add modality sections
    create_modality_sections(ax, angles, benchmarks)
    
    # Add legend and title
    add_legend_and_title(fig, ax, plotted_methods)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(OUTPUT_DIR) / "clam_radar_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Also save as PDF for publication
    output_path_pdf = Path(OUTPUT_DIR) / "clam_radar_plot.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Radar plot saved to:")
    print(f"  PNG: {output_path}")
    print(f"  PDF: {output_path_pdf}")
    
    # Show summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    
    for method in radar_df.columns:
        values = radar_df[method]
        non_zero_values = values[values > 0]
        if len(non_zero_values) > 0:
            print(f"{method}:")
            print(f"  Mean: {non_zero_values.mean():.1f}%")
            print(f"  Coverage: {len(non_zero_values)}/{len(radar_df)} benchmarks")
            print(f"  Range: {non_zero_values.min():.1f}% - {non_zero_values.max():.1f}%")
            print()
    
    plt.show()

if __name__ == "__main__":
    main()