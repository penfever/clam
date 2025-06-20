#!/usr/bin/env python3
"""
CLAM Results Manager Demonstration

This script demonstrates how to use the new unified results management system.
Shows examples of saving, loading, and managing experiment results across
different modalities (tabular, vision, audio).
"""

import sys
import os
import logging
import tempfile
import json
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from clam.utils.results_manager import (
    get_results_manager, 
    ExperimentMetadata, 
    EvaluationResults, 
    ResultsArtifacts
)
from clam.utils.results_migration import migrate_legacy_results, validate_result_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_usage():
    """Demonstrate basic results saving and loading."""
    print("\n" + "="*60)
    print("BASIC USAGE DEMONSTRATION")
    print("="*60)
    
    # Get the results manager
    results_manager = get_results_manager()
    
    # Create sample evaluation results
    results = EvaluationResults(
        accuracy=0.8542,
        precision_macro=0.8234,
        recall_macro=0.8456,
        f1_macro=0.8344,
        completion_rate=0.98,
        total_prediction_time=15.6,
        status="completed"
    )
    
    # Create experiment metadata
    metadata = ExperimentMetadata(
        model_name="tabllm",
        dataset_id="adult",
        modality="tabular",
        num_samples_train=1000,
        num_samples_test=200,
        num_classes=2,
        class_names=["<=50K", ">50K"],
        k_shot=5,
        random_seed=42,
        training_time_seconds=45.2
    )
    
    # Save the results
    print("Saving experiment results...")
    experiment_dir = results_manager.save_evaluation_results(
        model_name="tabllm",
        dataset_id="adult",
        modality="tabular",
        results=results,
        experiment_metadata=metadata
    )
    
    print(f"Results saved to: {experiment_dir}")
    
    # Load the results back
    print("\nLoading experiment results...")
    loaded_results = results_manager.load_evaluation_results(
        model_name="tabllm",
        dataset_id="adult",
        modality="tabular"
    )
    
    if loaded_results:
        print("Successfully loaded results!")
        print(f"Accuracy: {loaded_results['results']['accuracy']}")
        print(f"Model: {loaded_results['metadata']['model_name']}")
        print(f"Dataset: {loaded_results['metadata']['dataset_id']}")
    else:
        print("Failed to load results")


def demo_multiple_modalities():
    """Demonstrate saving results for different modalities."""
    print("\n" + "="*60)
    print("MULTIPLE MODALITIES DEMONSTRATION")
    print("="*60)
    
    results_manager = get_results_manager()
    
    # Tabular experiment
    print("Saving tabular experiment...")
    tabular_results = EvaluationResults(
        accuracy=0.8542,
        f1_macro=0.8344,
        status="completed"
    )
    tabular_metadata = ExperimentMetadata(
        model_name="jolt",
        dataset_id="diabetes",
        modality="tabular",
        k_shot=10
    )
    
    results_manager.save_evaluation_results(
        model_name="jolt",
        dataset_id="diabetes", 
        modality="tabular",
        results=tabular_results,
        experiment_metadata=tabular_metadata
    )
    
    # Vision experiment
    print("Saving vision experiment...")
    vision_results = EvaluationResults(
        accuracy=0.7234,
        status="completed",
        visualization_paths=["tsne_plot.png", "knn_plot.png"]
    )
    vision_metadata = ExperimentMetadata(
        model_name="clam_tsne",
        dataset_id="cifar10",
        modality="vision",
        model_config={
            "use_3d_tsne": True,
            "knn_k": 5,
            "vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct"
        }
    )
    vision_artifacts = ResultsArtifacts(
        visualizations={"main_plot": "tsne_visualization.png"},
        plots=["plot1.png", "plot2.png"]
    )
    
    results_manager.save_evaluation_results(
        model_name="clam_tsne",
        dataset_id="cifar10",
        modality="vision", 
        results=vision_results,
        experiment_metadata=vision_metadata,
        artifacts=vision_artifacts
    )
    
    # Audio experiment
    print("Saving audio experiment...")
    audio_results = EvaluationResults(
        accuracy=0.6789,
        status="completed"
    )
    audio_metadata = ExperimentMetadata(
        model_name="clam_audio",
        dataset_id="esc50",
        modality="audio",
        model_config={
            "embedding_model": "whisper",
            "whisper_model": "large-v2",
            "audio_duration": 5.0
        }
    )
    
    results_manager.save_evaluation_results(
        model_name="clam_audio",
        dataset_id="esc50",
        modality="audio",
        results=audio_results,
        experiment_metadata=audio_metadata
    )
    
    print("All experiments saved successfully!")


def demo_listing_experiments():
    """Demonstrate listing and querying experiments."""
    print("\n" + "="*60)
    print("LISTING EXPERIMENTS DEMONSTRATION")
    print("="*60)
    
    results_manager = get_results_manager()
    
    # List all experiments
    print("All experiments:")
    all_experiments = results_manager.list_experiments()
    for exp in all_experiments:
        print(f"  {exp['modality']}/{exp['dataset_id']}/{exp['model_name']}")
    
    # List by modality
    print("\nTabular experiments:")
    tabular_experiments = results_manager.list_experiments(modality="tabular")
    for exp in tabular_experiments:
        print(f"  {exp['dataset_id']}/{exp['model_name']}")
    
    # List by dataset
    print("\nExperiments on adult dataset:")
    adult_experiments = results_manager.list_experiments(dataset_id="adult")
    for exp in adult_experiments:
        print(f"  {exp['modality']}/{exp['model_name']}")


def demo_summary_report():
    """Demonstrate generating summary reports."""
    print("\n" + "="*60)
    print("SUMMARY REPORT DEMONSTRATION")
    print("="*60)
    
    results_manager = get_results_manager()
    
    # Generate comprehensive report
    print("Generating summary report...")
    report = results_manager.create_summary_report()
    
    print(f"Total experiments: {report['summary']['total_experiments']}")
    print(f"Modalities: {', '.join(report['summary']['modalities'])}")
    print(f"Datasets: {', '.join(report['summary']['datasets'])}")
    print(f"Models: {', '.join(report['summary']['models'])}")
    
    # Show best results
    if report['experiments']:
        successful_experiments = [
            exp for exp in report['experiments']
            if exp.get('accuracy') is not None
        ]
        
        if successful_experiments:
            best_exp = max(successful_experiments, key=lambda x: x['accuracy'])
            print(f"\nBest result: {best_exp['accuracy']:.4f} accuracy")
            print(f"  Model: {best_exp['model_name']}")
            print(f"  Dataset: {best_exp['dataset_id']}")
            print(f"  Modality: {best_exp['modality']}")


def demo_artifacts_management():
    """Demonstrate artifact management."""
    print("\n" + "="*60)
    print("ARTIFACTS MANAGEMENT DEMONSTRATION")
    print("="*60)
    
    results_manager = get_results_manager()
    
    # Create temporary files to simulate artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some fake artifact files
        plot_file = Path(temp_dir) / "visualization.png"
        plot_file.write_text("fake plot data")
        
        data_file = Path(temp_dir) / "predictions.csv"
        data_file.write_text("prediction,true_label\n0,0\n1,1\n")
        
        print(f"Created temporary artifacts in: {temp_dir}")
        
        # Save experiment with artifacts
        results = EvaluationResults(accuracy=0.8765, status="completed")
        metadata = ExperimentMetadata(
            model_name="test_model",
            dataset_id="test_dataset", 
            modality="vision"
        )
        
        experiment_dir = results_manager.save_evaluation_results(
            model_name="test_model",
            dataset_id="test_dataset",
            modality="vision",
            results=results,
            experiment_metadata=metadata
        )
        
        # Now save additional artifacts
        print("Saving additional artifacts...")
        artifact_paths = results_manager.save_artifacts(
            model_name="test_model",
            dataset_id="test_dataset",
            modality="vision",
            artifacts={
                "main_plot": str(plot_file),
                "predictions": str(data_file)
            },
            copy_files=True
        )
        
        print("Artifact paths:")
        for name, path in artifact_paths.items():
            print(f"  {name}: {path}")


def demo_backward_compatibility():
    """Demonstrate backward compatibility with existing scripts."""
    print("\n" + "="*60)
    print("BACKWARD COMPATIBILITY DEMONSTRATION")
    print("="*60)
    
    # Simulate legacy result format
    legacy_result = {
        'model_name': 'legacy_model',
        'dataset_name': 'legacy_dataset',
        'accuracy': 0.8234,
        'precision_macro': 0.8100,
        'recall_macro': 0.8300,
        'f1_macro': 0.8200,
        'training_time': 67.8,
        'prediction_time': 12.3,
        'num_test_samples': 500,
        'completion_rate': 0.96
    }
    
    # Use the unified save function with backward compatibility
    from clam.utils.json_utils import save_results
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Saving legacy format to: {temp_dir}")
        
        # Save using legacy method
        save_results([legacy_result], temp_dir, "legacy_dataset")
        
        # List files created
        legacy_files = list(Path(temp_dir).rglob("*.json"))
        print(f"Created {len(legacy_files)} files:")
        for file_path in legacy_files:
            print(f"  {file_path}")
        
        # Validate the created file
        if legacy_files:
            validation_result = validate_result_file(str(legacy_files[0]))
            print(f"Validation result: {validation_result['status']}")
            if validation_result['status'] == 'valid':
                print(f"  Detected format: {validation_result['format_type']}")


def demo_migration():
    """Demonstrate migrating legacy results."""
    print("\n" + "="*60)
    print("MIGRATION DEMONSTRATION")
    print("="*60)
    
    # Create some legacy result files to migrate
    with tempfile.TemporaryDirectory() as temp_dir:
        legacy_dir = Path(temp_dir) / "legacy_results"
        legacy_dir.mkdir()
        
        # Create legacy result files
        legacy_results = [
            {
                'model_name': 'old_tabllm',
                'dataset_name': 'wine',
                'accuracy': 0.8567,
                'training_time': 45.2,
                'prediction_time': 8.9
            },
            {
                'model_name': 'old_jolt', 
                'dataset_name': 'iris',
                'accuracy': 0.9234,
                'r2_score': None,
                'training_time': 23.1
            }
        ]
        
        for i, result in enumerate(legacy_results):
            result_file = legacy_dir / f"{result['model_name']}_results.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        print(f"Created legacy results in: {legacy_dir}")
        
        # Migrate them (dry run)
        print("Running migration (dry run)...")
        stats = migrate_legacy_results(str(legacy_dir), dry_run=True)
        
        print(f"Migration stats:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Would migrate: {stats['successful']}")
        print(f"  Would fail: {stats['failed']}")


def main():
    """Run all demonstrations."""
    print("CLAM Results Manager Demonstration")
    print("This demo shows how to use the unified results management system.")
    
    try:
        demo_basic_usage()
        demo_multiple_modalities()
        demo_listing_experiments()
        demo_summary_report()
        demo_artifacts_management()
        demo_backward_compatibility()
        demo_migration()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("All demonstrations completed successfully!")
        print("Check the results directory for saved experiments.")
        
        # Show final state
        results_manager = get_results_manager()
        experiments = results_manager.list_experiments()
        print(f"\nTotal experiments created: {len(experiments)}")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())