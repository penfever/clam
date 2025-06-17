#!/usr/bin/env python3
"""
Transform cc18_semantic datasets to tabarena_semantic format supersets.

This script converts cc18_semantic JSON files to be supersets of the tabarena_semantic format,
supporting both classification (target_classes) and regression (target_variable) formats.

Usage:
    python transform_cc18_to_tabarena_complete.py <input_dir> <output_dir> [--dry-run] [--verbose]
"""

import json
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class CC18ToTabarenaTransformer:
    def __init__(self):
        # Required fields for tabarena classification format
        self.required_classification_fields = {
            'dataset_name', 'description', 'original_source', 
            'columns', 'target_classes', 'dataset_history', 'inference_notes'
        }
        
        # Required fields for tabarena regression format
        self.required_regression_fields = {
            'dataset_name', 'description', 'original_source', 
            'columns', 'target_variable', 'dataset_history', 'inference_notes'
        }
        
        self.stats = {
            'total_files': 0,
            'already_compatible': 0,
            'successfully_transformed': 0,
            'failed_transformations': 0,
            'empty_files': 0,
            'classification_datasets': 0,
            'regression_datasets': 0,
            'skipped_files': []
        }

    def determine_dataset_type(self, data: Dict[str, Any]) -> str:
        """Determine if dataset is classification or regression."""
        # Check target_type first
        if 'target_type' in data:
            target_type = data['target_type'].lower()
            if target_type in ['continuous', 'numeric', 'real', 'regression']:
                return 'regression'
            elif target_type in ['binary', 'multiclass', 'classification', 'categorical']:
                return 'classification'
        
        # Check target_variable structure
        if 'target_variable' in data:
            target_var = data['target_variable']
            if isinstance(target_var, dict):
                var_type = target_var.get('type', '').lower()
                if var_type in ['continuous', 'numeric', 'real']:
                    return 'regression'
                elif 'classes' in target_var:
                    return 'classification'
        
        # Default to classification if target_classes exist or target_values look categorical
        if 'target_classes' in data:
            return 'classification'
        
        if 'target_values' in data:
            target_values = data['target_values']
            if isinstance(target_values, dict):
                # If all keys look like class labels (strings, small numbers), likely classification
                keys = list(target_values.keys())
                if all(isinstance(k, str) or (isinstance(k, (int, float)) and abs(k) < 100) for k in keys):
                    return 'classification'
            elif isinstance(target_values, list) and len(target_values) < 20:
                return 'classification'
        
        # Default to classification
        return 'classification'

    def check_already_compatible(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if data already matches tabarena format and return type."""
        dataset_type = self.determine_dataset_type(data)
        
        if dataset_type == 'classification':
            required_fields = self.required_classification_fields
            # Check if all required top-level fields exist
            if not all(field in data for field in required_fields):
                return False, dataset_type
            
            # Check target_classes structure
            classes = data.get('target_classes', [])
            required_class_fields = {'name', 'meaning'}
            if not isinstance(classes, list) or not classes:
                return False, dataset_type
            if not all(isinstance(cls, dict) and all(field in cls for field in required_class_fields) for cls in classes):
                return False, dataset_type
        else:  # regression
            required_fields = self.required_regression_fields
            # Check if all required top-level fields exist
            if not all(field in data for field in required_fields):
                return False, dataset_type
            
            # Check target_variable structure
            target_var = data.get('target_variable', {})
            required_var_fields = {'name', 'type', 'description'}
            if not isinstance(target_var, dict) or not all(field in target_var for field in required_var_fields):
                return False, dataset_type
        
        # Check original_source structure (common to both)
        source = data.get('original_source', {})
        required_source_fields = {'creator', 'institution', 'date', 'publication'}
        if not isinstance(source, dict) or not all(field in source for field in required_source_fields):
            return False, dataset_type
        
        # Check columns structure (common to both)
        columns = data.get('columns', [])
        required_column_fields = {'name', 'semantic_description', 'data_type'}
        if not isinstance(columns, list) or not columns:
            return False, dataset_type
        if not all(isinstance(col, dict) and all(field in col for field in required_column_fields) for col in columns):
            return False, dataset_type
        
        return True, dataset_type

    def transform_dataset_name(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract dataset_name from various patterns."""
        if 'dataset_name' in data:
            return str(data['dataset_name'])
        elif 'dataset' in data:
            return str(data['dataset'])
        return None

    def transform_description(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract description from various patterns."""
        if 'description' in data:
            return str(data['description'])
        elif 'dataset_description' in data:
            return str(data['dataset_description'])
        return None

    def transform_original_source(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract or construct original_source from various patterns."""
        # Already has original_source
        if 'original_source' in data and isinstance(data['original_source'], dict):
            source = data['original_source'].copy()
            # Ensure all required fields exist
            defaults = {
                'creator': 'Unknown',
                'institution': 'Unknown', 
                'date': 'Unknown',
                'publication': 'Unknown'
            }
            for field in ['creator', 'institution', 'date', 'publication']:
                if field not in source:
                    source[field] = defaults[field]
            return source
        
        # Construct from other fields
        source = {}
        
        # Creator mapping
        if 'creator' in data:
            if isinstance(data['creator'], dict):
                creator_name = data['creator'].get('name', 'Unknown')
                source['creator'] = str(creator_name)
            else:
                source['creator'] = str(data['creator'])
        elif 'donor' in data:
            source['creator'] = str(data['donor'])
        else:
            source['creator'] = 'Unknown'
        
        # Institution mapping  
        if 'institution' in data:
            source['institution'] = str(data['institution'])
        elif 'source' in data:
            source['institution'] = str(data['source'])
        elif 'creator' in data and isinstance(data['creator'], dict):
            source['institution'] = data['creator'].get('affiliation', 'Unknown')
        else:
            source['institution'] = 'Unknown'
        
        # Date mapping
        if 'date' in data:
            source['date'] = str(data['date'])
        elif 'date_donated' in data:
            source['date'] = str(data['date_donated'])
        elif 'year' in data:
            source['date'] = str(data['year'])
        else:
            source['date'] = 'Unknown'
        
        # Publication mapping
        if 'publication' in data:
            source['publication'] = str(data['publication'])
        elif 'citations' in data and isinstance(data['citations'], list) and data['citations']:
            # Extract first citation
            first_citation = data['citations'][0]
            if isinstance(first_citation, dict):
                title = first_citation.get('title', 'Unknown')
                author = first_citation.get('author', 'Unknown')
                source['publication'] = f"{author}: {title}"
            else:
                source['publication'] = str(first_citation)
        else:
            source['publication'] = 'Unknown'
        
        return source

    def transform_columns(self, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Extract or construct columns from various patterns."""
        # Already has columns in correct format
        if 'columns' in data and isinstance(data['columns'], list):
            columns = []
            for col in data['columns']:
                if isinstance(col, dict) and all(field in col for field in ['name', 'semantic_description', 'data_type']):
                    columns.append(col)
            if columns:
                return columns
        
        # Pattern: feature_names + feature_descriptions (both as dicts)
        if 'feature_names' in data and isinstance(data['feature_names'], dict):
            feature_names = data['feature_names']
            feature_descriptions = data.get('feature_descriptions', {})
            feature_types = data.get('feature_types', {})
            
            columns = []
            for internal_name, display_name in feature_names.items():
                col = {'name': str(display_name)}
                
                # Get description
                if internal_name in feature_descriptions:
                    col['semantic_description'] = str(feature_descriptions[internal_name])
                elif display_name in feature_descriptions:
                    col['semantic_description'] = str(feature_descriptions[display_name])
                else:
                    col['semantic_description'] = f"Feature: {display_name}"
                
                # Get data type
                if isinstance(feature_types, dict):
                    if 'all' in feature_types:
                        col['data_type'] = str(feature_types['all'])
                    elif internal_name in feature_types:
                        col['data_type'] = str(feature_types[internal_name])
                    elif display_name in feature_types:
                        col['data_type'] = str(feature_types[display_name])
                    else:
                        col['data_type'] = 'unknown'
                else:
                    col['data_type'] = 'unknown'
                
                columns.append(col)
            return columns
        
        # Pattern: feature_names as list + feature_descriptions as dict
        if 'feature_names' in data and isinstance(data['feature_names'], list):
            feature_names = data['feature_names']
            feature_descriptions = data.get('feature_descriptions', {})
            feature_types = data.get('feature_types', {})
            
            columns = []
            for name in feature_names:
                col = {'name': str(name)}
                
                # Get description
                if isinstance(feature_descriptions, dict) and name in feature_descriptions:
                    col['semantic_description'] = str(feature_descriptions[name])
                elif isinstance(feature_descriptions, list) and len(feature_descriptions) > len(columns):
                    col['semantic_description'] = str(feature_descriptions[len(columns)])
                else:
                    col['semantic_description'] = f"Feature: {name}"
                
                # Get data type
                if isinstance(feature_types, dict):
                    if 'all' in feature_types:
                        col['data_type'] = str(feature_types['all'])
                    elif name in feature_types:
                        col['data_type'] = str(feature_types[name])
                    else:
                        col['data_type'] = 'unknown'
                else:
                    col['data_type'] = 'unknown'
                
                columns.append(col)
            return columns
        
        # Pattern: feature_description (singular) with feature count
        if 'feature_description' in data and 'features' in data:
            try:
                num_features = int(data['features'])
                feature_desc = data['feature_description']
                feature_types = data.get('feature_types', {})
                
                # Determine base data type
                if isinstance(feature_types, dict):
                    base_type = feature_types.get('all', 'numeric')
                else:
                    base_type = 'numeric'
                
                columns = []
                for i in range(num_features):
                    if isinstance(feature_desc, dict):
                        # Feature description is a dict with structure info
                        desc = feature_desc.get('description', f"Feature {i+1}")
                        col = {
                            'name': f"feature_{i+1}",
                            'semantic_description': desc,
                            'data_type': base_type
                        }
                    else:
                        # Feature description is a string
                        col = {
                            'name': f"feature_{i+1}",
                            'semantic_description': f"{feature_desc} (feature {i+1})",
                            'data_type': base_type
                        }
                    columns.append(col)
                return columns
            except (ValueError, TypeError):
                pass
        
        # Pattern: input_features dict with metadata
        if 'input_features' in data and isinstance(data['input_features'], dict):
            input_features = data['input_features']
            
            # Get number of features
            num_features = None
            if 'number_of_features' in input_features:
                try:
                    num_features = int(input_features['number_of_features'])
                except (ValueError, TypeError):
                    pass
            
            # Get feature description and types
            feature_desc = input_features.get('feature_description', 'Feature')
            feature_types = input_features.get('feature_types', 'unknown')
            
            if num_features:
                columns = []
                for i in range(num_features):
                    col = {
                        'name': f"feature_{i+1}",
                        'semantic_description': f"{feature_desc} (feature {i+1})",
                        'data_type': str(feature_types)
                    }
                    columns.append(col)
                return columns
        
        # Pattern: feature_categories + feature_examples
        if 'feature_categories' in data and 'feature_examples' in data and 'features' in data:
            try:
                num_features = int(data['features'])
                feature_categories = data['feature_categories']
                feature_examples = data['feature_examples']
                feature_types = data.get('feature_types', {})
                
                # Determine base data type
                if isinstance(feature_types, dict):
                    if 'all' in feature_types:
                        base_type = str(feature_types['all'])
                    else:
                        base_type = 'mixed'
                else:
                    base_type = 'unknown'
                
                columns = []
                for i in range(num_features):
                    # Create a generic feature description based on categories
                    if feature_categories:
                        categories_desc = ", ".join(feature_categories.keys())
                        desc = f"Feature {i+1} from categories: {categories_desc}"
                    else:
                        desc = f"Feature {i+1}"
                    
                    col = {
                        'name': f"feature_{i+1}",
                        'semantic_description': desc,
                        'data_type': base_type
                    }
                    columns.append(col)
                return columns
            except (ValueError, TypeError):
                pass
        
        # Pattern: feature_structure + feature_descriptions
        if 'feature_structure' in data and 'feature_descriptions' in data:
            feature_structure = data['feature_structure']
            feature_descriptions = data['feature_descriptions']
            feature_types = data.get('feature_types', {})
            
            # Determine base data type
            if isinstance(feature_types, dict):
                base_type = feature_types.get('all', 'unknown')
            else:
                base_type = 'unknown'
            
            columns = []
            
            # Extract feature counts from structure
            for feature_type, count_or_desc in feature_structure.items():
                if isinstance(count_or_desc, int):
                    count = count_or_desc
                    type_desc = feature_descriptions.get(feature_type, f"{feature_type} features")
                    
                    for i in range(count):
                        col = {
                            'name': f"{feature_type}_feature_{i+1}",
                            'semantic_description': f"{type_desc} (feature {i+1})",
                            'data_type': base_type
                        }
                        columns.append(col)
            
            if columns:
                return columns
        
        return None

    def transform_target_variable(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract or construct target_variable for regression datasets."""
        # Check if target_variable already exists
        if 'target_variable' in data and isinstance(data['target_variable'], dict):
            target_var = data['target_variable'].copy()
            # Ensure required fields exist
            if 'name' not in target_var:
                target_var['name'] = data.get('target', 'target')
            if 'type' not in target_var:
                target_var['type'] = 'continuous'
            if 'description' not in target_var:
                target_var['description'] = f"Target variable: {target_var['name']}"
            return target_var
        
        # Construct from other fields
        target_var = {}
        
        # Get target name
        if 'target' in data:
            target_var['name'] = str(data['target'])
        else:
            target_var['name'] = 'target'
        
        # Set type
        target_var['type'] = 'continuous'
        
        # Get description
        if 'target_description' in data:
            target_var['description'] = str(data['target_description'])
        else:
            target_var['description'] = f"Continuous target variable: {target_var['name']}"
        
        return target_var

    def transform_target_classes(self, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Extract or construct target_classes from various patterns."""
        # Already has target_classes
        if 'target_classes' in data and isinstance(data['target_classes'], list):
            classes = []
            for cls in data['target_classes']:
                if isinstance(cls, dict) and all(field in cls for field in ['name', 'meaning']):
                    classes.append(cls)
            if classes:
                return classes
        
        # Pattern: target_values as dict (most common in cc18_semantic)
        if 'target_values' in data and isinstance(data['target_values'], dict):
            classes = []
            for class_name, class_meaning in data['target_values'].items():
                classes.append({
                    'name': str(class_name),
                    'meaning': str(class_meaning) if class_meaning else f"Class {class_name}"
                })
            return classes
        
        # Pattern: target_variable dict with values list (for classification)
        if 'target_variable' in data and isinstance(data['target_variable'], dict):
            target_var = data['target_variable']
            if 'values' in target_var and isinstance(target_var['values'], list):
                classes = []
                target_desc = target_var.get('description', 'Target class')
                for class_val in target_var['values']:
                    classes.append({
                        'name': str(class_val),
                        'meaning': f"{target_desc}: {class_val}"
                    })
                return classes
        
        # Pattern: target_values as list
        if 'target_values' in data and isinstance(data['target_values'], list):
            classes = []
            class_dist = data.get('class_distribution', {})
            
            for target_val in data['target_values']:
                class_info = {'name': str(target_val)}
                
                # Try to get meaning from class distribution or other sources
                if isinstance(class_dist, dict) and target_val in class_dist:
                    class_info['meaning'] = f"Class {target_val} (frequency: {class_dist[target_val]})"
                else:
                    class_info['meaning'] = f"Target class: {target_val}"
                
                classes.append(class_info)
            return classes
        
        # Extract from target_variable if it's classification
        if 'target_variable' in data:
            target_var = data['target_variable']
            if isinstance(target_var, dict):
                target_name = target_var.get('name', 'target')
                target_desc = target_var.get('description', f"Target variable: {target_name}")
                
                # Try to infer classes if this is classification
                if 'classes' in target_var:
                    classes = []
                    for cls in target_var['classes']:
                        classes.append({
                            'name': str(cls),
                            'meaning': f"Class {cls} for {target_desc}"
                        })
                    return classes
        
        return None

    def transform_dataset_history(self, data: Dict[str, Any]) -> str:
        """Extract or construct dataset_history."""
        # Direct mapping
        if 'dataset_history' in data:
            return str(data['dataset_history'])
        
        # Construct from other fields
        history_parts = []
        
        if 'data_collection' in data:
            history_parts.append(f"Data Collection: {data['data_collection']}")
        
        if 'research_context' in data:
            history_parts.append(f"Research Context: {data['research_context']}")
        
        if 'background' in data:
            history_parts.append(f"Background: {data['background']}")
        
        if history_parts:
            return ". ".join(history_parts)
        
        dataset_name = self.transform_dataset_name(data) or "Unknown Dataset"
        return f"Dataset history not available for {dataset_name}"

    def transform_inference_notes(self, data: Dict[str, Any]) -> str:
        """Extract or construct inference_notes."""
        # Direct mapping
        if 'inference_notes' in data:
            return str(data['inference_notes'])
        
        # Construct from other fields
        notes_parts = []
        
        if 'challenges' in data:
            notes_parts.append(f"Challenges: {data['challenges']}")
        
        if 'advantages' in data:
            notes_parts.append(f"Advantages: {data['advantages']}")
        
        if 'research_applications' in data:
            notes_parts.append(f"Applications: {data['research_applications']}")
        
        if 'use_case' in data:
            notes_parts.append(f"Use Case: {data['use_case']}")
        
        if notes_parts:
            return ". ".join(notes_parts)
        
        dataset_name = self.transform_dataset_name(data) or "Unknown Dataset"
        return f"Inference notes not available for {dataset_name}"

    def transform_dataset(self, data: Dict[str, Any], filename: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Transform a single dataset to tabarena format."""
        issues = []
        
        # Handle empty files
        if not data:
            issues.append("Empty file")
            return None, issues
        
        # Determine dataset type
        dataset_type = self.determine_dataset_type(data)
        
        # Check if already compatible
        is_compatible, detected_type = self.check_already_compatible(data)
        if is_compatible:
            logger.info(f"{filename}: Already compatible with tabarena {detected_type} format")
            return data, []
        
        # Start with a copy of original data to preserve everything
        transformed = data.copy()
        
        # Transform common fields
        dataset_name = self.transform_dataset_name(data)
        if not dataset_name:
            issues.append("No dataset name found (missing 'dataset_name' or 'dataset')")
            return None, issues
        transformed['dataset_name'] = dataset_name
        
        description = self.transform_description(data)
        if not description:
            issues.append("No description found")
            return None, issues
        transformed['description'] = description
        
        original_source = self.transform_original_source(data)
        if not original_source:
            issues.append("Cannot construct original_source")
            return None, issues
        transformed['original_source'] = original_source
        
        columns = self.transform_columns(data)
        if not columns:
            issues.append("Cannot extract column/feature information")
            return None, issues
        transformed['columns'] = columns
        
        # Transform target based on dataset type
        if dataset_type == 'classification':
            target_classes = self.transform_target_classes(data)
            if not target_classes:
                issues.append("Cannot extract target classes for classification dataset")
                return None, issues
            transformed['target_classes'] = target_classes
            # Remove target_variable if it exists (for clean classification format)
            if 'target_variable' in transformed:
                del transformed['target_variable']
        else:  # regression
            target_variable = self.transform_target_variable(data)
            if not target_variable:
                issues.append("Cannot extract target variable for regression dataset")
                return None, issues
            transformed['target_variable'] = target_variable
            # Remove target_classes if it exists (for clean regression format)
            if 'target_classes' in transformed:
                del transformed['target_classes']
        
        transformed['dataset_history'] = self.transform_dataset_history(data)
        transformed['inference_notes'] = self.transform_inference_notes(data)
        
        logger.info(f"{filename}: Successfully transformed to tabarena {dataset_type} format")
        return transformed, []

    def process_directory(self, input_dir: str, output_dir: str, dry_run: bool = False) -> Dict[str, Any]:
        """Process all JSON files in input directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if not dry_run:
            output_path.mkdir(parents=True, exist_ok=True)
        
        json_files = list(input_path.glob("*.json"))
        self.stats['total_files'] = len(json_files)
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        failed_files = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check for empty files
                if not data:
                    self.stats['empty_files'] += 1
                    logger.warning(f"{json_file.name}: Empty file, skipping")
                    continue
                
                transformed_data, issues = self.transform_dataset(data, json_file.name)
                
                if transformed_data is None:
                    self.stats['failed_transformations'] += 1
                    failed_info = {
                        'filename': json_file.name,
                        'issues': issues
                    }
                    failed_files.append(failed_info)
                    self.stats['skipped_files'].append(failed_info)
                    logger.warning(f"{json_file.name}: Transformation failed - {', '.join(issues)}")
                else:
                    # Determine if this was already compatible or transformed
                    is_compatible, detected_type = self.check_already_compatible(data)
                    
                    if is_compatible:
                        self.stats['already_compatible'] += 1
                    else:
                        self.stats['successfully_transformed'] += 1
                    
                    # Count by type
                    final_type = self.determine_dataset_type(transformed_data)
                    if final_type == 'classification':
                        self.stats['classification_datasets'] += 1
                    else:
                        self.stats['regression_datasets'] += 1
                    
                    if not dry_run:
                        output_file = output_path / json_file.name
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(transformed_data, f, indent=2, ensure_ascii=False)
                        logger.info(f"{json_file.name}: Saved to {output_file}")
            
            except json.JSONDecodeError as e:
                self.stats['failed_transformations'] += 1
                failed_info = {
                    'filename': json_file.name,
                    'issues': [f"JSON decode error: {e}"]
                }
                failed_files.append(failed_info)
                self.stats['skipped_files'].append(failed_info)
                logger.error(f"{json_file.name}: JSON decode error - {e}")
            
            except Exception as e:
                self.stats['failed_transformations'] += 1
                failed_info = {
                    'filename': json_file.name,
                    'issues': [f"Unexpected error: {e}"]
                }
                failed_files.append(failed_info)
                self.stats['skipped_files'].append(failed_info)
                logger.error(f"{json_file.name}: Unexpected error - {e}")
        
        return {
            'stats': self.stats,
            'failed_files': failed_files
        }

    def print_summary(self, results: Dict[str, Any]):
        """Print transformation summary."""
        stats = results['stats']
        failed_files = results['failed_files']
        
        print("\n" + "="*60)
        print("TRANSFORMATION SUMMARY")
        print("="*60)
        print(f"Total files processed: {stats['total_files']}")
        print(f"Already compatible: {stats['already_compatible']}")
        print(f"Successfully transformed: {stats['successfully_transformed']}")
        print(f"Failed transformations: {stats['failed_transformations']}")
        print(f"Empty files: {stats['empty_files']}")
        
        total_success = stats['already_compatible'] + stats['successfully_transformed']
        if stats['total_files'] > 0:
            success_rate = (total_success / stats['total_files']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        print(f"\nDataset Types:")
        print(f"Classification datasets: {stats['classification_datasets']}")
        print(f"Regression datasets: {stats['regression_datasets']}")
        
        if failed_files:
            print(f"\nFAILED FILES ({len(failed_files)}):")
            print("-" * 40)
            for failure in failed_files:
                print(f"• {failure['filename']}")
                for issue in failure['issues']:
                    print(f"  - {issue}")
        
        print("\nTransformation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Transform cc18_semantic datasets to tabarena_semantic format supersets"
    )
    parser.add_argument(
        "input_dir",
        help="Input directory containing cc18_semantic JSON files"
    )
    parser.add_argument(
        "output_dir", 
        help="Output directory for transformed JSON files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze files without writing output"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    transformer = CC18ToTabarenaTransformer()
    
    try:
        results = transformer.process_directory(
            args.input_dir,
            args.output_dir,
            dry_run=args.dry_run
        )
        transformer.print_summary(results)
        
        # Exit with non-zero code if there were failures
        if results['stats']['failed_transformations'] > 0:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()