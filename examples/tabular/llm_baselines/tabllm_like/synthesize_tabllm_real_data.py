#!/usr/bin/env python3
"""
Script to synthesize TabLLM notes from actual OpenML CC18 data.
This script loads real datasets and creates notes from actual data rows.
"""

import json
import os
import glob
import yaml
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

# Optional: Import OpenML if available
try:
    import openml
    OPENML_AVAILABLE = True
except ImportError:
    OPENML_AVAILABLE = False
    print("Warning: OpenML not installed. Will generate synthetic examples only.")

# Define paths (dynamically resolve based on script location)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
SEMANTIC_DIR = os.path.join(project_root, "data", "cc18_semantic_complete")
OUTPUT_DIR = current_dir
NOTES_PER_DATASET = 100  # Number of example notes to generate per dataset


def load_openml_dataset(dataset_id: int) -> Optional[pd.DataFrame]:
    """Load a dataset from OpenML."""
    if not OPENML_AVAILABLE:
        return None
    
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=attribute_names)
        df['target'] = y
        
        return df
    except Exception as e:
        print(f"Error loading OpenML dataset {dataset_id}: {e}")
        return None


def generate_note_from_row(row: pd.Series, semantic_info: Dict[str, Any], exclude_target: bool = True) -> str:
    """Generate a TabLLM-style note from a data row."""
    note_parts = []
    
    # Get feature descriptions based on the structure of semantic JSON
    if 'columns' in semantic_info:
        # Structure like kr-vs-kp dataset
        for col in semantic_info['columns']:
            col_name = col['name']
            if col_name in row.index and (not exclude_target or col_name != 'target'):
                semantic_desc = col['semantic_description']
                value = row[col_name]
                
                # Handle different data types
                if pd.isna(value):
                    note_parts.append(f"The {semantic_desc} is missing.")
                elif isinstance(value, (int, float)):
                    if value == int(value):
                        note_parts.append(f"The {semantic_desc} is {int(value)}.")
                    else:
                        note_parts.append(f"The {semantic_desc} is {value:.2f}.")
                else:
                    note_parts.append(f"The {semantic_desc} is {value}.")
    
    elif 'feature_descriptions' in semantic_info:
        # Structure like letter dataset
        for feat_name, feat_desc in semantic_info['feature_descriptions'].items():
            if feat_name in row.index and (not exclude_target or feat_name != 'target'):
                value = row[feat_name]
                # Clean up description for better readability
                clean_desc = feat_desc.replace(' (integer)', '').replace(' (float)', '')
                
                if pd.isna(value):
                    note_parts.append(f"The {clean_desc} is missing.")
                elif isinstance(value, (int, float)):
                    if value == int(value):
                        note_parts.append(f"The {clean_desc} is {int(value)}.")
                    else:
                        note_parts.append(f"The {clean_desc} is {value:.2f}.")
                else:
                    note_parts.append(f"The {clean_desc} is {value}.")
    
    elif 'feature_description' in semantic_info:
        # Handle adult dataset format (singular feature_description)
        for feat_name, feat_desc in semantic_info['feature_description'].items():
            if feat_name in row.index and (not exclude_target or feat_name != 'target'):
                value = row[feat_name]
                # Create more natural descriptions
                feature_label = feat_name.replace('-', ' ').replace('_', ' ')
                
                if pd.isna(value):
                    note_parts.append(f"The {feature_label} is missing.")
                elif isinstance(value, (int, float)):
                    if value == int(value):
                        note_parts.append(f"The {feature_label} is {int(value)}.")
                    else:
                        note_parts.append(f"The {feature_label} is {value:.2f}.")
                else:
                    note_parts.append(f"The {feature_label} is {value}.")
    
    elif 'feature_names' in semantic_info:
        # Fallback to feature_names if descriptions not available
        for feat_name, feat_meaning in semantic_info['feature_names'].items():
            if feat_name in row.index and (not exclude_target or feat_name != 'target'):
                value = row[feat_name]
                
                if pd.isna(value):
                    note_parts.append(f"The {feat_meaning} is missing.")
                elif isinstance(value, (int, float)):
                    if value == int(value):
                        note_parts.append(f"The {feat_meaning} is {int(value)}.")
                    else:
                        note_parts.append(f"The {feat_meaning} is {value:.2f}.")
                else:
                    note_parts.append(f"The {feat_meaning} is {value}.")
    
    return " ".join(note_parts)


def process_dataset_to_notes(json_file: str, max_notes: int = NOTES_PER_DATASET) -> List[Dict[str, Any]]:
    """Process a single dataset and generate notes."""
    notes_data = []
    
    try:
        with open(json_file, 'r') as f:
            semantic_info = json.load(f)
        
        # Extract dataset info
        dataset_name = semantic_info.get('dataset_name', semantic_info.get('dataset', Path(json_file).stem))
        dataset_id = int(Path(json_file).stem)
        
        print(f"Processing dataset {dataset_name} (ID: {dataset_id})...")
        
        # Try to load actual data
        df = load_openml_dataset(dataset_id)
        
        if df is not None and len(df) > 0:
            # Sample rows for notes
            n_samples = min(max_notes, len(df))
            sampled_df = df.sample(n=n_samples, random_state=42)
            
            for idx, row in sampled_df.iterrows():
                note = generate_note_from_row(row, semantic_info, exclude_target=True)
                target = row.get('target', row.get('class', None))
                
                notes_data.append({
                    'dataset': dataset_name,
                    'dataset_id': dataset_id,
                    'note': note,
                    'target': target,
                    'row_index': idx
                })
        
        else:
            # Generate synthetic examples if real data not available
            print(f"  Generating synthetic examples for {dataset_name}...")
            
            for i in range(min(10, max_notes)):  # Generate fewer synthetic examples
                example_features = {}
                
                if 'columns' in semantic_info:
                    for col in semantic_info['columns']:
                        col_name = col['name']
                        data_type = col.get('data_type', 'unknown')
                        if 'binary' in data_type:
                            example_features[col_name] = np.random.choice(['yes', 'no'])
                        elif 'categorical' in data_type:
                            if '(' in data_type and ')' in data_type:
                                values = data_type[data_type.find('(')+1:data_type.find(')')].split('/')
                                example_features[col_name] = np.random.choice(values)
                            else:
                                example_features[col_name] = f'category_{np.random.randint(1, 4)}'
                        else:
                            example_features[col_name] = np.random.randint(0, 100)
                
                elif 'feature_description' in semantic_info:
                    # Handle adult dataset format
                    for feat_name, feat_desc in semantic_info['feature_description'].items():
                        if feat_desc.lower() == 'continuous':
                            if 'age' in feat_name.lower():
                                example_features[feat_name] = np.random.randint(17, 90)
                            elif 'fnlwgt' in feat_name.lower():
                                example_features[feat_name] = np.random.randint(10000, 1000000)
                            elif 'education-num' in feat_name.lower():
                                example_features[feat_name] = np.random.randint(1, 16)
                            elif 'capital' in feat_name.lower():
                                example_features[feat_name] = np.random.randint(0, 10000)
                            elif 'hours' in feat_name.lower():
                                example_features[feat_name] = np.random.randint(1, 80)
                            else:
                                example_features[feat_name] = np.random.randint(0, 100)
                        elif ',' in feat_desc:
                            # Categorical with comma-separated values
                            values = [v.strip() for v in feat_desc.split(',')]
                            example_features[feat_name] = np.random.choice(values)
                        else:
                            example_features[feat_name] = f'value_{np.random.randint(1, 5)}'
                
                elif 'feature_names' in semantic_info or 'feature_descriptions' in semantic_info:
                    feature_list = semantic_info.get('feature_descriptions', semantic_info.get('feature_names', {}))
                    for feat_name in feature_list.keys():
                        example_features[feat_name] = np.random.randint(0, 16)  # Based on letter dataset scale
                
                # Create Series for note generation
                row_series = pd.Series(example_features)
                note = generate_note_from_row(row_series, semantic_info, exclude_target=True)
                
                # Generate synthetic target
                if 'target_classes' in semantic_info:
                    target = np.random.choice([tc['name'] for tc in semantic_info['target_classes']])
                elif 'target_values' in semantic_info:
                    target = np.random.choice(list(semantic_info['target_values'].keys()))
                else:
                    target = np.random.choice(['0', '1'])
                
                notes_data.append({
                    'dataset': dataset_name,
                    'dataset_id': dataset_id,
                    'note': note,
                    'target': target,
                    'row_index': i,
                    'synthetic': True
                })
        
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
    
    return notes_data


def create_notes_bank_files():
    """Create note bank files for all datasets."""
    # Get all JSON files
    json_files = glob.glob(os.path.join(SEMANTIC_DIR, "*.json"))
    
    all_notes = []
    dataset_notes = {}  # Store notes by dataset
    
    for json_file in sorted(json_files):
        notes = process_dataset_to_notes(json_file)
        if notes:
            dataset_name = notes[0]['dataset']
            dataset_notes[dataset_name] = notes
            all_notes.extend(notes)
    
    # Write individual dataset note files
    for dataset_name, notes in dataset_notes.items():
        notes_filename = f"notes_{dataset_name}.jsonl"
        notes_path = os.path.join(OUTPUT_DIR, "notes", notes_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(notes_path), exist_ok=True)
        
        with open(notes_path, 'w') as f:
            for note_item in notes:
                f.write(json.dumps(note_item) + '\n')
        
        # Also create simple text version
        text_filename = f"notes_{dataset_name}.txt"
        text_path = os.path.join(OUTPUT_DIR, "notes", text_filename)
        
        with open(text_path, 'w') as f:
            for note_item in notes:
                f.write(f"['{note_item['note']}'] -> {note_item['target']}\n")
    
    # Write combined notes file
    combined_path = os.path.join(OUTPUT_DIR, "notes", "all_notes.jsonl")
    with open(combined_path, 'w') as f:
        for note_item in all_notes:
            f.write(json.dumps(note_item) + '\n')
    
    print(f"\nCreated note files for {len(dataset_notes)} datasets")
    print(f"Total notes generated: {len(all_notes)}")
    print(f"Notes directory: {os.path.join(OUTPUT_DIR, 'notes')}")


def generate_note_from_semantic_info(semantic_info: Dict[str, Any]) -> str:
    """Generate a note example from semantic information."""
    try:
        # Extract feature descriptions from semantic info
        features = []
        if 'columns' in semantic_info:
            for col in semantic_info['columns']:
                if col.get('name') != 'target':
                    name = col.get('name', 'feature')
                    desc = col.get('semantic_description', name)
                    # Create a synthetic example value
                    if 'type' in col and col['type'] == 'categorical':
                        value = 'example_category'
                    else:
                        value = '42'
                    features.append(f"The {desc} is {value}")
        elif 'feature_descriptions' in semantic_info:
            for name, desc in semantic_info['feature_descriptions'].items():
                features.append(f"The {desc} is example_value")
        elif 'feature_description' in semantic_info:
            for name, desc in semantic_info['feature_description'].items():
                features.append(f"The {desc} is example_value")
        else:
            # Fallback
            features = ["The feature is example_value"]
        
        return ". ".join(features[:10])  # Limit to first 10 features
        
    except Exception as e:
        # Fallback note
        return "This is an example tabular data instance with various features"


def create_template_for_dataset(semantic_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create a template for a dataset from semantic information."""
    try:
        # Extract class information
        classes = []
        class_descriptions = {}
        
        if 'target_description' in semantic_info:
            target_desc = semantic_info['target_description']
            if 'class_names' in semantic_info:
                classes = semantic_info['class_names']
            elif 'classes' in semantic_info:
                classes = semantic_info['classes']
            else:
                classes = ['class_0', 'class_1']  # Default binary
            
            # Create class descriptions
            for cls in classes:
                class_descriptions[str(cls)] = f"{cls} ({target_desc})"
        else:
            # Fallback
            classes = ['class_0', 'class_1']
            class_descriptions = {'class_0': 'Class 0', 'class_1': 'Class 1'}
        
        # Create template structure
        template_data = {
            'templates': {
                'default': {
                    'jinja': 'Given the following information about a data instance, which of the following classes does this instance belong to: {{ answer_choices | join(", ") }}? {{ text }}',
                    'answer_choices': ' ||| '.join(classes),
                    'class_descriptions': class_descriptions
                }
            }
        }
        
        return template_data
        
    except Exception as e:
        # Fallback template
        return {
            'templates': {
                'default': {
                    'jinja': 'Given the following information about a data instance, which of the following classes does this instance belong to: {{ answer_choices | join(", ") }}? {{ text }}',
                    'answer_choices': 'class_0 ||| class_1',
                    'class_descriptions': {'class_0': 'Class 0', 'class_1': 'Class 1'}
                }
            }
        }


def main():
    """Main function."""
    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "notes"), exist_ok=True)
    
    print("Synthesizing TabLLM notes from OpenML CC18 datasets...")
    create_notes_bank_files()
    print("\nDone!")


if __name__ == "__main__":
    main()