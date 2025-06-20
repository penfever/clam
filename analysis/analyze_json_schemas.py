#!/usr/bin/env python3
"""
JSON Schema Analysis Tool

This script analyzes JSON files in a directory to identify schema patterns,
group files by schema similarity, and provide detailed schema analysis.

Usage:
    python analyze_json_schemas.py <directory_path> [--output <output_file>]

Author: Claude Code Assistant
"""

import json
import os
import sys
import argparse
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict, Counter
import hashlib


def get_schema_signature(obj: Any, path: str = "") -> Dict[str, str]:
    """
    Extract schema signature from a JSON object.
    Returns a dictionary mapping field paths to their types.
    """
    schema = {}
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                schema[current_path] = "object"
                schema.update(get_schema_signature(value, current_path))
            elif isinstance(value, list):
                schema[current_path] = "array"
                if value:  # If array is not empty, analyze first element
                    array_element_schema = get_schema_signature(value[0], f"{current_path}[*]")
                    schema.update(array_element_schema)
            else:
                value_type = type(value).__name__
                schema[current_path] = value_type
    
    elif isinstance(obj, list):
        schema[path] = "array"
        if obj:  # If array is not empty, analyze first element
            array_element_schema = get_schema_signature(obj[0], f"{path}[*]")
            schema.update(array_element_schema)
    else:
        value_type = type(obj).__name__
        schema[path] = value_type
    
    return schema


def schema_to_hash(schema: Dict[str, str]) -> str:
    """Convert schema dictionary to a hash for grouping."""
    # Sort keys to ensure consistent hashing
    sorted_items = sorted(schema.items())
    schema_string = json.dumps(sorted_items, sort_keys=True)
    return hashlib.md5(schema_string.encode()).hexdigest()


def analyze_json_files(directory_path: str) -> Dict[str, Any]:
    """
    Analyze all JSON files in the given directory.
    
    Returns:
        Dictionary containing analysis results
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    if not json_files:
        return {
            "total_files": 0,
            "error": "No JSON files found in directory"
        }
    
    results = {
        "directory": directory_path,
        "total_files": len(json_files),
        "successful_parses": 0,
        "failed_parses": 0,
        "parse_errors": [],
        "schema_groups": defaultdict(list),
        "schema_signatures": {},
        "schema_samples": {},
        "unique_schemas": 0,
        "field_frequency": Counter(),
        "summary": {}
    }
    
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract schema signature
            schema = get_schema_signature(data)
            schema_hash = schema_to_hash(schema)
            
            # Group files by schema
            results["schema_groups"][schema_hash].append(filename)
            results["schema_signatures"][schema_hash] = schema
            
            # Store a sample for each schema (first file encountered)
            if schema_hash not in results["schema_samples"]:
                # Store a truncated sample of the data
                sample_data = {}
                for key, value in data.items():
                    if isinstance(value, (dict, list)) and len(str(value)) > 200:
                        sample_data[key] = f"<{type(value).__name__} with {len(value)} items>"
                    else:
                        sample_data[key] = value
                results["schema_samples"][schema_hash] = {
                    "sample_file": filename,
                    "sample_data": sample_data
                }
            
            # Count field frequencies
            for field_path in schema.keys():
                results["field_frequency"][field_path] += 1
            
            results["successful_parses"] += 1
            
        except json.JSONDecodeError as e:
            results["failed_parses"] += 1
            results["parse_errors"].append({
                "file": filename,
                "error": str(e)
            })
        except Exception as e:
            results["failed_parses"] += 1
            results["parse_errors"].append({
                "file": filename,
                "error": f"Unexpected error: {str(e)}"
            })
    
    # Calculate summary statistics
    results["unique_schemas"] = len(results["schema_groups"])
    
    # Convert defaultdict to regular dict for JSON serialization
    results["schema_groups"] = dict(results["schema_groups"])
    
    # Create summary
    results["summary"] = {
        "total_files": results["total_files"],
        "successful_parses": results["successful_parses"],
        "failed_parses": results["failed_parses"],
        "unique_schemas": results["unique_schemas"],
        "largest_schema_group": max(len(files) for files in results["schema_groups"].values()) if results["schema_groups"] else 0,
        "most_common_fields": results["field_frequency"].most_common(10)
    }
    
    return results


def print_analysis_report(results: Dict[str, Any]) -> None:
    """Print a formatted analysis report."""
    print("=" * 80)
    print("JSON SCHEMA ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"\nDirectory: {results['directory']}")
    print(f"Total JSON files: {results['total_files']}")
    print(f"Successfully parsed: {results['successful_parses']}")
    print(f"Parse failures: {results['failed_parses']}")
    print(f"Unique schemas found: {results['unique_schemas']}")
    
    if results["parse_errors"]:
        print(f"\nParse Errors:")
        for error in results["parse_errors"]:
            print(f"  - {error['file']}: {error['error']}")
    
    print(f"\n" + "=" * 50)
    print("SCHEMA GROUPS")
    print("=" * 50)
    
    # Sort schema groups by number of files (largest first)
    sorted_groups = sorted(results["schema_groups"].items(), 
                          key=lambda x: len(x[1]), reverse=True)
    
    for i, (schema_hash, files) in enumerate(sorted_groups, 1):
        print(f"\nSchema Group {i} (Hash: {schema_hash[:8]}...)")
        print(f"Files: {len(files)}")
        print(f"Files list: {', '.join(sorted(files))}")
        
        schema = results["schema_signatures"][schema_hash]
        print(f"Schema structure ({len(schema)} fields):")
        
        # Group fields by depth for better readability
        root_fields = []
        nested_fields = []
        
        for field_path, field_type in sorted(schema.items()):
            if '.' not in field_path and '[*]' not in field_path:
                root_fields.append((field_path, field_type))
            else:
                nested_fields.append((field_path, field_type))
        
        # Print root fields first
        for field_path, field_type in root_fields:
            print(f"  {field_path}: {field_type}")
        
        # Print nested fields
        if nested_fields:
            print("  Nested fields:")
            for field_path, field_type in nested_fields:
                indent = "    " + "  " * (field_path.count('.') + field_path.count('[*]'))
                clean_path = field_path.split('.')[-1]
                print(f"{indent}{clean_path}: {field_type}")
        
        # Show sample data
        if schema_hash in results["schema_samples"]:
            sample = results["schema_samples"][schema_hash]
            print(f"Sample from {sample['sample_file']}:")
            sample_keys = list(sample['sample_data'].keys())[:5]  # Show first 5 keys
            for key in sample_keys:
                value = sample['sample_data'][key]
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {key}: {value}")
            if len(sample['sample_data']) > 5:
                print(f"  ... and {len(sample['sample_data']) - 5} more fields")
    
    print(f"\n" + "=" * 50)
    print("FIELD FREQUENCY ANALYSIS")
    print("=" * 50)
    
    print(f"\nMost common fields across all files:")
    for field, count in results["field_frequency"].most_common(15):
        percentage = (count / results["successful_parses"]) * 100
        print(f"  {field}: {count} files ({percentage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze JSON schemas in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_json_schemas.py /path/to/json/files
  python analyze_json_schemas.py /path/to/json/files --output analysis_report.json
        """
    )
    
    parser.add_argument(
        "directory", 
        help="Path to directory containing JSON files"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Optional output file to save detailed results as JSON"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only output summary, no detailed report"
    )
    
    args = parser.parse_args()
    
    try:
        results = analyze_json_files(args.directory)
        
        if not args.quiet:
            print_analysis_report(results)
        else:
            summary = results["summary"]
            print(f"Files: {summary['total_files']}, "
                  f"Schemas: {summary['unique_schemas']}, "
                  f"Success: {summary['successful_parses']}, "
                  f"Errors: {summary['failed_parses']}")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {args.output}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()