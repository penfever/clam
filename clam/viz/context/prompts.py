"""
Prompt generation for multi-visualization reasoning contexts.

This module generates sophisticated prompts that guide VLM systems to reason
effectively across multiple visualization perspectives.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PromptGenerator:
    """
    Generate reasoning prompts for multi-visualization contexts.
    
    This class creates detailed prompts that help VLM systems understand
    and reason across multiple visualization perspectives of the same data.
    """
    
    def __init__(self, composition_config):
        """
        Initialize the prompt generator.
        
        Args:
            composition_config: CompositionConfig object
        """
        self.config = composition_config
        self.logger = logging.getLogger(__name__)
    
    def generate_prompt(
        self,
        visualizations: List[Dict[str, Any]],
        highlight_indices: Optional[List[int]] = None,
        custom_context: Optional[str] = None,
        task_description: Optional[str] = None,
        data_shape: Optional[tuple] = None,
        n_classes: Optional[int] = None
    ) -> str:
        """
        Generate a comprehensive reasoning prompt using enhanced VLM utilities.
        
        Args:
            visualizations: List of visualization information dictionaries
            highlight_indices: Indices of highlighted points
            custom_context: Additional context about the data
            task_description: Specific task to perform
            data_shape: Shape of the original data
            n_classes: Number of classes (for classification)
            
        Returns:
            Generated prompt string
        """
        # Import the enhanced VLM prompting utilities
        from clam.utils.vlm_prompting import create_classification_prompt, create_regression_prompt
        
        # Prepare multi_viz_info for the enhanced utilities
        multi_viz_info = []
        for viz in visualizations:
            multi_viz_info.append({
                'method': viz.get('method', 'Unknown'),
                'description': viz.get('description', f"Visualization of data using {viz.get('method', 'unknown')} method")
            })
        
        # Determine if this is classification or regression
        is_regression = any(viz.get('config', {}).get('task_type') == 'regression' for viz in visualizations)
        
        # Build dataset description
        dataset_description = ""
        if data_shape:
            dataset_description += f"Dataset with {data_shape[0]} samples and {data_shape[1]} features. "
        
        if custom_context:
            dataset_description += f"{custom_context} "
            
        if task_description:
            dataset_description += f"Task: {task_description}"
        
        if is_regression:
            # For regression, we need target stats - use dummy values if not available
            target_stats = {
                'min': 0.0,
                'max': 1.0, 
                'mean': 0.5,
                'std': 0.3
            }
            
            # Try to extract target stats from visualizations
            for viz in visualizations:
                if 'target_stats' in viz:
                    target_stats = viz['target_stats']
                    break
            
            return create_regression_prompt(
                target_stats=target_stats,
                modality="tabular",  # Default to tabular
                dataset_description=dataset_description,
                multi_viz_info=multi_viz_info
            )
        else:
            # For classification
            if n_classes:
                class_names = [f"Class {i}" for i in range(n_classes)]
            else:
                class_names = ["Class 0", "Class 1", "Class 2"]  # Default
            
            return create_classification_prompt(
                class_names=class_names,
                modality="tabular",  # Default to tabular
                dataset_description=dataset_description,
                use_semantic_names=False,  # Use Class X format for consistency
                multi_viz_info=multi_viz_info
            )
    
    def _generate_introduction(
        self,
        n_visualizations: int,
        data_shape: Optional[tuple],
        n_classes: Optional[int]
    ) -> str:
        """Generate the introduction section of the prompt."""
        
        intro = f"""You are analyzing a dataset through {n_visualizations} different visualization perspectives. Each visualization reveals different aspects of the underlying data structure and relationships."""
        
        if data_shape:
            intro += f" The original dataset contains {data_shape[0]} samples with {data_shape[1]} features."
        
        if n_classes:
            intro += f" This is a classification task with {n_classes} classes."
        
        intro += " Your goal is to synthesize insights from all visualizations to provide a comprehensive understanding of the data patterns."
        
        return intro
    
    def _generate_visualization_descriptions(
        self,
        visualizations: List[Dict[str, Any]]
    ) -> str:
        """Generate descriptions for each visualization method."""
        
        descriptions = ["**Visualization Methods:**"]
        
        for i, viz_info in enumerate(visualizations, 1):
            method = viz_info['method']
            description = viz_info.get('description', '')
            config = viz_info.get('config', {})
            
            desc_text = f"{i}. **{method.upper()}**: {description}"
            
            # Add method-specific context
            if method.lower() == 'tsne':
                desc_text += " t-SNE preserves local neighborhood structures and is excellent for revealing clusters and local patterns."
            elif method.lower() == 'umap':
                desc_text += " UMAP preserves both local and global structure, often showing clearer cluster separation than t-SNE."
            elif method.lower() == 'pca':
                desc_text += " PCA shows linear relationships and the directions of maximum variance in the data."
            elif 'spectral' in method.lower():
                desc_text += " Spectral embedding reveals the manifold structure using graph-based relationships."
            elif 'lle' in method.lower() or 'locally' in method.lower():
                desc_text += " Locally Linear Embedding reconstructs the local geometry of the data manifold."
            elif method.lower() == 'isomap':
                desc_text += " Isomap preserves geodesic distances along the data manifold."
            elif method.lower() == 'mds':
                desc_text += " MDS preserves pairwise distances between data points in the reduced space."
            
            # Add 3D information if applicable
            if config.get('use_3d'):
                desc_text += " This visualization uses 3D representation for enhanced spatial understanding."
            
            descriptions.append(desc_text)
        
        return "\n".join(descriptions)
    
    def _generate_cross_references(
        self,
        visualizations: List[Dict[str, Any]]
    ) -> str:
        """Generate cross-references between visualization methods."""
        
        if len(visualizations) < 2:
            return ""
        
        cross_refs = ["**Cross-Visualization Analysis:**"]
        
        # Compare preservation properties
        linear_methods = [v for v in visualizations if v['method'].lower() in ['pca']]
        nonlinear_methods = [v for v in visualizations if v['method'].lower() in ['tsne', 'umap', 'isomap', 'lle']]
        
        if linear_methods and nonlinear_methods:
            cross_refs.append(
                "- Compare the linear perspective (PCA) with nonlinear methods to understand if the data has inherent nonlinear structure."
            )
        
        # Compare local vs global preservation
        local_methods = [v for v in visualizations if v['method'].lower() in ['tsne', 'lle']]
        global_methods = [v for v in visualizations if v['method'].lower() in ['umap', 'isomap', 'mds']]
        
        if local_methods and global_methods:
            cross_refs.append(
                "- Local structure preserving methods (t-SNE, LLE) may show different cluster arrangements than global methods (UMAP, Isomap, MDS)."
            )
        
        # Clustering consistency
        cross_refs.append(
            "- Look for consistent cluster patterns across methods - clusters that appear in multiple visualizations are likely genuine data structures."
        )
        
        # Outlier detection
        cross_refs.append(
            "- Points that appear as outliers in multiple visualizations are likely true outliers in the data."
        )
        
        # Method-specific insights
        method_pairs = [
            ('tsne', 'umap', "t-SNE may show tighter clusters while UMAP preserves more global structure"),
            ('pca', 'tsne', "PCA shows linear separability while t-SNE reveals nonlinear cluster structure"),
            ('isomap', 'lle', "Isomap preserves geodesic distances while LLE focuses on local linearity"),
        ]
        
        for method1, method2, insight in method_pairs:
            methods_present = [v['method'].lower() for v in visualizations]
            if method1 in methods_present and method2 in methods_present:
                cross_refs.append(f"- {insight}.")
        
        return "\n".join(cross_refs)
    
    def _generate_highlight_analysis(self, highlight_indices: List[int]) -> str:
        """Generate analysis guidance for highlighted points."""
        
        n_highlighted = len(highlight_indices)
        
        analysis = f"""**Highlighted Points Analysis:**
You should pay special attention to the {n_highlighted} highlighted point{"s" if n_highlighted > 1 else ""} (marked in red or with special symbols). For these points:

- Analyze their position across all visualizations
- Look for consistency in their neighborhood relationships
- Identify if they represent outliers, cluster centers, or boundary cases
- Consider how different methods position these points relative to the overall data structure
- Note any visualization-specific patterns for these highlighted points"""
        
        return analysis
    
    def _generate_reasoning_guidance(self) -> str:
        """Generate reasoning guidance based on the configuration focus."""
        
        focus = self.config.reasoning_focus
        
        guidance_map = {
            "comparison": """**Reasoning Focus - Comparison:**
Compare and contrast the patterns shown in each visualization:
- Which methods show similar cluster structures?
- Where do the methods disagree, and what might this indicate?
- Are there patterns visible in some methods but not others?
- How do the relative positions of clusters change across methods?""",
            
            "consensus": """**Reasoning Focus - Consensus:**
Look for patterns that are consistent across multiple visualizations:
- Which clusters appear in most or all visualizations?
- What data structures are reliably preserved across methods?
- Which relationships between data points are method-independent?
- What can you confidently conclude about the data structure?""",
            
            "divergence": """**Reasoning Focus - Divergence:**
Focus on where the visualizations show different patterns:
- Which methods reveal unique perspectives on the data?
- Where do visualizations disagree about cluster boundaries or outliers?
- What might cause these differences (method assumptions, parameter settings)?
- How can these differences inform our understanding of the data complexity?"""
        }
        
        return guidance_map.get(focus, guidance_map["comparison"])
    
    def _generate_final_instructions(self, visualizations: List[Dict[str, Any]]) -> str:
        """Generate final analysis instructions."""
        
        instructions = """**Analysis Instructions:**

Please provide a comprehensive analysis that includes:

1. **Overall Data Structure**: Describe the general patterns you observe across all visualizations
2. **Cluster Analysis**: Identify and describe any clusters, their characteristics, and consistency across methods
3. **Outlier Detection**: Point out any outliers and explain how they appear across different visualizations
4. **Method-Specific Insights**: Highlight what each visualization method reveals that others might miss
5. **Synthesis**: Provide a unified interpretation that reconciles the different perspectives
6. **Confidence Assessment**: Indicate which patterns you're most confident about and which require further investigation

Be specific about spatial relationships, use directional terms (left, right, upper, lower), and reference the colored patterns or symbols you observe."""
        
        # Add task-specific guidance
        if any(viz['config'].get('task_type') == 'regression' for viz in visualizations):
            instructions += "\n\n**Regression Task Note**: Pay attention to how the continuous target values are distributed spatially in each visualization. Look for gradients, smooth transitions, or discrete jumps in the target variable across the embedded space."
        
        return instructions
    
    def generate_comparison_prompt(
        self,
        method_pairs: List[tuple],
        specific_questions: Optional[List[str]] = None
    ) -> str:
        """
        Generate a prompt focused on comparing specific visualization methods.
        
        Args:
            method_pairs: List of (method1, method2) tuples to compare
            specific_questions: Optional list of specific comparison questions
            
        Returns:
            Comparison-focused prompt
        """
        prompt_parts = [
            "**Visualization Method Comparison Analysis**",
            "",
            f"You are comparing {len(method_pairs)} pairs of visualization methods applied to the same dataset.",
            "Focus on understanding how different methods reveal different aspects of the data structure.",
            ""
        ]
        
        for i, (method1, method2) in enumerate(method_pairs, 1):
            prompt_parts.append(f"**Comparison {i}: {method1.upper()} vs {method2.upper()}**")
            prompt_parts.append(self._get_method_comparison_guidance(method1, method2))
            prompt_parts.append("")
        
        if specific_questions:
            prompt_parts.append("**Specific Questions to Address:**")
            for i, question in enumerate(specific_questions, 1):
                prompt_parts.append(f"{i}. {question}")
            prompt_parts.append("")
        
        prompt_parts.append(self._generate_final_instructions([]))
        
        return "\n".join(prompt_parts)
    
    def _get_method_comparison_guidance(self, method1: str, method2: str) -> str:
        """Get specific guidance for comparing two methods."""
        
        comparisons = {
            ('tsne', 'umap'): "Compare cluster tightness and global structure preservation. UMAP often shows better global structure while t-SNE emphasizes local neighborhoods.",
            ('pca', 'tsne'): "Compare linear vs nonlinear structure. PCA shows linear separability while t-SNE reveals nonlinear patterns.",
            ('umap', 'pca'): "Compare nonlinear manifold structure (UMAP) with linear projections (PCA) to understand data complexity.",
            ('isomap', 'lle'): "Compare geodesic distance preservation (Isomap) with local linear reconstruction (LLE).",
            ('tsne', 'spectral'): "Compare neighborhood preservation (t-SNE) with graph-based manifold learning (Spectral).",
            ('mds', 'pca'): "Compare distance preservation (MDS) with variance maximization (PCA).",
        }
        
        # Try both orders
        key = (method1.lower(), method2.lower())
        reverse_key = (method2.lower(), method1.lower())
        
        guidance = comparisons.get(key) or comparisons.get(reverse_key)
        
        if guidance:
            return guidance
        else:
            return f"Compare how {method1} and {method2} represent the data structure differently. Look for similarities and differences in cluster arrangements, outlier detection, and overall data organization."