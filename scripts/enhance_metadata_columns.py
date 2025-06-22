#!/usr/bin/env python
"""
Enhance metadata columns with meaningful semantic descriptions.

This script analyzes all metadata JSON files in cc18_semantic and improves
the "columns" field by extracting semantic information from other fields
like "feature_description", "feature_names", "feature_descriptions", etc.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import shutil
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataEnhancer:
    """Enhances metadata JSON files with better column descriptions."""
    
    def __init__(self, metadata_dir: Path, backup: bool = True):
        self.metadata_dir = Path(metadata_dir)
        self.backup = backup
        self.enhancement_stats = {
            'files_processed': 0,
            'files_enhanced': 0,
            'files_already_good': 0,
            'files_with_errors': 0
        }
        
    def backup_file(self, filepath: Path) -> None:
        """Create a backup of the original file."""
        if not self.backup:
            return
            
        backup_dir = filepath.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{filepath.stem}_{timestamp}.json"
        shutil.copy2(filepath, backup_path)
        logger.debug(f"Backed up {filepath.name} to {backup_path}")
        
    def extract_feature_info(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract feature information from various metadata fields.
        
        Returns list of dicts with 'name', 'description', 'data_type' keys.
        """
        features = []
        
        # Strategy 1: Use feature_description field (like 23.json)
        if 'feature_description' in data:
            feature_desc = data['feature_description']
            if isinstance(feature_desc, dict):
                # Dictionary format: feature name -> description
                for i, (name, desc) in enumerate(feature_desc.items()):
                    data_type = self._infer_data_type(desc)
                    features.append({
                        'name': name,
                        'description': desc,
                        'data_type': data_type
                    })
            elif isinstance(feature_desc, str):
                # String format: general description for all features
                num_features = data.get('features', 0)
                data_type = self._infer_data_type_from_context(data)
                for i in range(num_features):
                    features.append({
                        'name': f"Fourier coefficient {i+1}" if "fourier" in feature_desc.lower() else f"Feature {i+1}",
                        'description': f"{feature_desc} (coefficient {i+1})" if "fourier" in feature_desc.lower() else f"{feature_desc} (feature {i+1})",
                        'data_type': data_type
                    })
        
        # Strategy 2: Use feature_names + feature_descriptions (like 53.json)
        elif 'feature_names' in data and 'feature_descriptions' in data:
            feature_names = data['feature_names']
            feature_descriptions = data['feature_descriptions']
            
            for key in feature_names:
                if key in feature_descriptions:
                    desc = feature_descriptions[key]
                else:
                    desc = feature_names[key]
                    
                data_type = self._infer_data_type(desc)
                features.append({
                    'name': feature_names[key],
                    'description': desc,
                    'data_type': data_type
                })
        
        # Strategy 3: Use feature_names only
        elif 'feature_names' in data:
            feature_names = data['feature_names']
            for key, name in feature_names.items():
                data_type = self._infer_data_type_from_context(data)
                features.append({
                    'name': name,
                    'description': name,  # Use name as description if no better info
                    'data_type': data_type
                })
        
        # Strategy 4: Generate from feature count (fallback)
        elif 'features' in data:
            num_features = data['features']
            data_type = self._infer_data_type_from_context(data)
            
            # Check if we have any feature type information
            if 'feature_descriptions' in data and 'all_features' in data['feature_descriptions']:
                base_desc = data['feature_descriptions']['all_features']
                for i in range(num_features):
                    features.append({
                        'name': f"feature_{i+1}",
                        'description': f"{base_desc} (feature {i+1})",
                        'data_type': data_type
                    })
            else:
                # Very basic fallback
                for i in range(num_features):
                    features.append({
                        'name': f"feature_{i+1}",
                        'description': f"Feature {i+1}",
                        'data_type': data_type
                    })
        
        return features
        
    def _infer_data_type(self, description) -> str:
        """Infer data type from feature description."""
        # Handle case where description might not be a string
        if not isinstance(description, str):
            description = str(description)
            
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['binary', '0=', '1=', 'yes/no', 't/f']):
            return 'binary'
        elif any(word in desc_lower for word in ['categorical', 'category', 'class']):
            return 'categorical'
        elif any(word in desc_lower for word in ['age', 'count', 'number', 'years']):
            return 'integer'
        elif any(word in desc_lower for word in ['continuous', 'ratio', 'measure', 'index']):
            return 'continuous'
        else:
            return 'numeric'
            
    def _infer_data_type_from_context(self, data: Dict[str, Any]) -> str:
        """Infer data type from overall dataset context."""
        if 'feature_types' in data:
            types = data['feature_types']
            if isinstance(types, dict):
                # Return the most common type
                if 'all' in types:
                    return types['all']
                elif 'continuous' in types:
                    return 'continuous'
                elif 'categorical' in types:
                    return 'categorical'
                else:
                    return 'numeric'
        return 'numeric'
        
    def assess_columns_quality(self, columns: List[Dict[str, Any]]) -> str:
        """
        Assess the quality of existing columns field.
        
        Returns: 'good', 'poor', 'missing'
        """
        if not columns:
            return 'missing'
            
        # Check if columns have meaningful names and descriptions
        good_indicators = 0
        total_columns = len(columns)
        
        for col in columns:
            name = col.get('name', '')
            desc = col.get('semantic_description', '')
            
            # Good indicators: 
            # - Name is not generic (not just "feature_N")
            # - Description is not just repeating the name
            # - Description has substantial content
            
            if not name.startswith('feature_') and len(name) > 3:
                good_indicators += 1
                
            if desc and desc != name and len(desc) > 10:
                good_indicators += 1
                
        # If more than 60% of columns have good indicators, consider it good
        quality_score = good_indicators / (total_columns * 2)  # 2 indicators per column
        
        if quality_score > 0.6:
            return 'good'
        elif quality_score > 0.2:
            return 'fair'
        else:
            return 'poor'
            
    def enhance_metadata_file(self, filepath: Path) -> bool:
        """
        Enhance a single metadata file.
        
        Returns True if file was enhanced, False if no changes made.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            original_data = json.dumps(data, sort_keys=True)
            
            # Assess current columns quality
            current_columns = data.get('columns', [])
            quality = self.assess_columns_quality(current_columns)
            
            logger.info(f"Processing {filepath.name}: current quality = {quality}")
            
            if quality == 'good':
                logger.info(f"  → Already has good column descriptions, skipping")
                self.enhancement_stats['files_already_good'] += 1
                return False
                
            # Extract feature information
            extracted_features = self.extract_feature_info(data)
            
            if not extracted_features:
                logger.warning(f"  → Could not extract meaningful feature information")
                return False
                
            # Create enhanced columns
            enhanced_columns = []
            for feature in extracted_features:
                enhanced_columns.append({
                    'name': feature['name'],
                    'semantic_description': feature['description'],
                    'data_type': feature['data_type']
                })
                
            # Update the data
            data['columns'] = enhanced_columns
            
            # Check if we actually made improvements
            new_data = json.dumps(data, sort_keys=True)
            if new_data == original_data:
                logger.info(f"  → No improvements possible")
                return False
                
            # Backup original file
            self.backup_file(filepath)
            
            # Write enhanced file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"  → Enhanced with {len(enhanced_columns)} feature descriptions")
            self.enhancement_stats['files_enhanced'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error processing {filepath.name}: {e}")
            self.enhancement_stats['files_with_errors'] += 1
            return False
            
    def enhance_all_files(self) -> None:
        """Enhance all JSON files in the metadata directory."""
        json_files = list(self.metadata_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        for filepath in sorted(json_files):
            self.enhancement_stats['files_processed'] += 1
            self.enhance_metadata_file(filepath)
            
        self.print_summary()
        
    def print_summary(self) -> None:
        """Print enhancement summary."""
        stats = self.enhancement_stats
        logger.info("\n" + "="*50)
        logger.info("ENHANCEMENT SUMMARY")
        logger.info("="*50)
        logger.info(f"Files processed:    {stats['files_processed']}")
        logger.info(f"Files enhanced:     {stats['files_enhanced']}")
        logger.info(f"Files already good: {stats['files_already_good']}")
        logger.info(f"Files with errors:  {stats['files_with_errors']}")
        logger.info("="*50)


def main():
    parser = argparse.ArgumentParser(description='Enhance metadata columns with semantic descriptions')
    parser.add_argument('--metadata_dir', 
                       default='data/cc18_semantic',
                       help='Directory containing metadata JSON files')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    metadata_dir = Path(args.metadata_dir)
    if not metadata_dir.exists():
        logger.error(f"Metadata directory not found: {metadata_dir}")
        return 1
        
    enhancer = MetadataEnhancer(metadata_dir, backup=not args.no_backup)
    
    if args.dry_run:
        logger.info("DRY RUN MODE - no files will be modified")
        # TODO: Implement dry run logic
    else:
        enhancer.enhance_all_files()
        
    return 0


if __name__ == '__main__':
    exit(main())