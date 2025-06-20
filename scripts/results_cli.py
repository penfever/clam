#!/usr/bin/env python3
"""
CLAM Results Management CLI

Command-line interface for managing CLAM experiment results.
Provides utilities for listing, migrating, validating, and cleaning up results.

Usage examples:
    # List all experiments
    python results_cli.py list
    
    # List experiments for specific modality
    python results_cli.py list --modality tabular
    
    # Migrate legacy results
    python results_cli.py migrate /path/to/legacy/results --dry-run
    
    # Generate summary report
    python results_cli.py report --output summary.json
    
    # Clean up old results
    python results_cli.py cleanup --days 30 --dry-run
    
    # Validate a specific result file
    python results_cli.py validate /path/to/result.json
"""

import argparse
import logging
import sys
import json
import os
from pathlib import Path
from typing import Optional

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from clam.utils.results_manager import get_results_manager
from clam.utils.results_migration import migrate_legacy_results, validate_result_file, ResultsMigrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_list(args):
    """List experiments."""
    results_manager = get_results_manager()
    
    experiments = results_manager.list_experiments(
        modality=args.modality,
        dataset_id=args.dataset_id,
        model_name=args.model_name
    )
    
    if not experiments:
        print("No experiments found matching the criteria.")
        return
    
    print(f"Found {len(experiments)} experiments:")
    print(f"{'Modality':<12} {'Dataset':<20} {'Model':<20} {'Path'}")
    print("-" * 80)
    
    for exp in experiments:
        print(f"{exp['modality']:<12} {exp['dataset_id']:<20} {exp['model_name']:<20} {exp['path']}")
    
    # Summary by modality
    if len(experiments) > 1:
        modalities = {}
        for exp in experiments:
            modality = exp['modality']
            modalities[modality] = modalities.get(modality, 0) + 1
        
        print(f"\nSummary by modality:")
        for modality, count in sorted(modalities.items()):
            print(f"  {modality}: {count} experiments")


def cmd_migrate(args):
    """Migrate legacy results."""
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory does not exist: {args.source_dir}")
        return 1
    
    print(f"Migrating results from: {args.source_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    stats = migrate_legacy_results(
        source_dir=args.source_dir,
        pattern=args.pattern,
        dry_run=args.dry_run
    )
    
    print("Migration Results:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped: {stats['skipped']}")
    
    if stats['errors']:
        print("\nErrors:")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    
    if args.dry_run:
        print("\nThis was a dry run. Use --no-dry-run to actually migrate files.")
    
    return 0 if stats['failed'] == 0 else 1


def cmd_report(args):
    """Generate summary report."""
    results_manager = get_results_manager()
    
    print("Generating summary report...")
    
    report = results_manager.create_summary_report(
        modality=args.modality,
        output_file=args.output
    )
    
    print(f"Summary Report:")
    print(f"  Total experiments: {report['summary']['total_experiments']}")
    print(f"  Modalities: {', '.join(report['summary']['modalities'])}")
    print(f"  Datasets: {len(report['summary']['datasets'])}")
    print(f"  Models: {len(report['summary']['models'])}")
    print(f"  Generated: {report['summary']['generated_at']}")
    
    if args.output:
        print(f"\nDetailed report saved to: {args.output}")
    
    # Show some sample results
    successful_experiments = [
        exp for exp in report['experiments'] 
        if exp.get('status') == 'completed' and exp.get('accuracy') is not None
    ]
    
    if successful_experiments:
        print(f"\nTop 10 results by accuracy:")
        top_results = sorted(
            successful_experiments, 
            key=lambda x: x.get('accuracy', 0), 
            reverse=True
        )[:10]
        
        print(f"{'Accuracy':<10} {'Model':<20} {'Dataset':<15} {'Modality':<10}")
        print("-" * 65)
        for exp in top_results:
            accuracy = exp.get('accuracy', 0)
            print(f"{accuracy:<10.4f} {exp['model_name']:<20} {exp['dataset_id']:<15} {exp['modality']:<10}")


def cmd_cleanup(args):
    """Clean up old results."""
    results_manager = get_results_manager()
    
    print(f"Cleaning up results older than {args.days} days...")
    print(f"Dry run: {args.dry_run}")
    print()
    
    stats = results_manager.cleanup_old_results(
        days_old=args.days,
        dry_run=args.dry_run
    )
    
    print("Cleanup Results:")
    print(f"  Scanned: {stats['scanned']} experiments")
    print(f"  Marked for deletion: {stats['marked_for_deletion']}")
    print(f"  Actually deleted: {stats['deleted']}")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB")
    print(f"  Freed space: {stats['freed_size_mb']:.1f} MB")
    print(f"  Errors: {stats['errors']}")
    
    if args.dry_run:
        print("\nThis was a dry run. Use --no-dry-run to actually delete files.")


def cmd_validate(args):
    """Validate a result file."""
    if not os.path.exists(args.file_path):
        print(f"Error: File does not exist: {args.file_path}")
        return 1
    
    print(f"Validating: {args.file_path}")
    
    validation_result = validate_result_file(args.file_path)
    
    if validation_result['status'] == 'valid':
        print("✓ File is valid")
        print(f"  Format: {validation_result['format_type']}")
        print(f"  Model: {validation_result['model_name']}")
        print(f"  Dataset: {validation_result['dataset_id']}")
        print(f"  Modality: {validation_result['modality']}")
        print(f"  Has accuracy: {validation_result['has_accuracy']}")
        print(f"  Has artifacts: {validation_result['has_artifacts']}")
        
        if validation_result['warnings']:
            print("Warnings:")
            for warning in validation_result['warnings']:
                print(f"  ⚠ {warning}")
        
        return 0
    else:
        print("✗ File is invalid")
        print(f"  Error: {validation_result['error']}")
        return 1


def cmd_info(args):
    """Show information about the results system."""
    results_manager = get_results_manager()
    
    print("CLAM Results Management System")
    print("=" * 40)
    print(f"Results base directory: {results_manager.get_results_base_dir()}")
    print(f"Supported modalities: {', '.join(results_manager._supported_modalities)}")
    
    # Count experiments by modality
    experiments = results_manager.list_experiments()
    modality_counts = {}
    for exp in experiments:
        modality = exp['modality']
        modality_counts[modality] = modality_counts.get(modality, 0) + 1
    
    print(f"\nCurrent experiments: {len(experiments)}")
    for modality, count in sorted(modality_counts.items()):
        print(f"  {modality}: {count}")
    
    # Show directory structure
    base_dir = results_manager.get_results_base_dir()
    if base_dir.exists():
        print(f"\nDirectory structure:")
        for modality_dir in sorted(base_dir.iterdir()):
            if modality_dir.is_dir():
                dataset_count = len([d for d in modality_dir.iterdir() if d.is_dir()])
                print(f"  {modality_dir.name}: {dataset_count} datasets")


def main():
    parser = argparse.ArgumentParser(
        description="CLAM Results Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list --modality tabular
  %(prog)s migrate /path/to/results --dry-run
  %(prog)s report --output summary.json
  %(prog)s cleanup --days 30 --dry-run
  %(prog)s validate result.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List experiments')
    list_parser.add_argument('--modality', help='Filter by modality')
    list_parser.add_argument('--dataset-id', help='Filter by dataset ID')
    list_parser.add_argument('--model-name', help='Filter by model name')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate legacy results')
    migrate_parser.add_argument('source_dir', help='Source directory containing legacy results')
    migrate_parser.add_argument('--pattern', default='*_results.json', help='File pattern to match')
    migrate_parser.add_argument('--dry-run', action='store_true', default=True, help='Only report what would be done')
    migrate_parser.add_argument('--no-dry-run', dest='dry_run', action='store_false', help='Actually perform migration')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate summary report')
    report_parser.add_argument('--modality', help='Filter by modality')
    report_parser.add_argument('--output', help='Output file for detailed report')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old results')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Remove results older than this many days')
    cleanup_parser.add_argument('--dry-run', action='store_true', default=True, help='Only report what would be deleted')
    cleanup_parser.add_argument('--no-dry-run', dest='dry_run', action='store_false', help='Actually delete files')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a result file')
    validate_parser.add_argument('file_path', help='Path to result file to validate')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command function
    command_functions = {
        'list': cmd_list,
        'migrate': cmd_migrate,
        'report': cmd_report,
        'cleanup': cmd_cleanup,
        'validate': cmd_validate,
        'info': cmd_info
    }
    
    try:
        return command_functions[args.command](args)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return 1
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())