#!/usr/bin/env python
"""
Example demonstrating the unified CLAM t-SNE classifier for all modalities.

This shows how to use the same ClamTsneClassifier class for tabular, audio, and vision data.
"""

from clam.models import ClamTsneClassifier, ClamAudioTsneClassifier, ClamImageTsneClassifier

# Example 1: Tabular data (default)
def tabular_example():
    """Example using CLAM for tabular data."""
    classifier = ClamTsneClassifier(
        modality="tabular",
        vlm_model_id="Qwen/Qwen2-VL-2B-Instruct",
        tsne_perplexity=30,
        use_semantic_names=True
    )
    
    # X_train, y_train = load_tabular_data()
    # classifier.fit(X_train, y_train)
    # predictions = classifier.predict(X_test)
    print("Tabular classifier initialized")


# Example 2: Audio data
def audio_example():
    """Example using CLAM for audio data."""
    # Option 1: Using the unified classifier with modality="audio"
    classifier = ClamTsneClassifier(
        modality="audio",
        vlm_model_id="Qwen/Qwen2-VL-2B-Instruct",
        embedding_model="whisper",  # or "clap"
        whisper_model="large-v2",
        include_spectrogram=True,
        tsne_perplexity=30
    )
    
    # Option 2: Using the convenience class (equivalent to above)
    classifier = ClamAudioTsneClassifier(
        vlm_model_id="Qwen/Qwen2-VL-2B-Instruct",
        embedding_model="whisper",
        whisper_model="large-v2",
        include_spectrogram=True,
        tsne_perplexity=30
    )
    
    # audio_files = ["path/to/audio1.wav", "path/to/audio2.wav", ...]
    # labels = [0, 1, ...]
    # classifier.fit(audio_files, labels)
    # predictions = classifier.predict(test_audio_files)
    print("Audio classifier initialized")


# Example 3: Vision/Image data
def vision_example():
    """Example using CLAM for vision/image data."""
    # Option 1: Using the unified classifier with modality="vision"
    classifier = ClamTsneClassifier(
        modality="vision",
        vlm_model_id="Qwen/Qwen2-VL-2B-Instruct",
        dinov2_model="dinov2_vitb14",
        tsne_perplexity=30,
        use_3d_tsne=False,
        use_knn_connections=True,
        nn_k=5
    )
    
    # Option 2: Using the convenience class (equivalent to above)
    classifier = ClamImageTsneClassifier(
        vlm_model_id="Qwen/Qwen2-VL-2B-Instruct",
        dinov2_model="dinov2_vitb14",
        tsne_perplexity=30,
        use_3d_tsne=False,
        use_knn_connections=True,
        nn_k=5
    )
    
    # image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
    # labels = [0, 1, ...]
    # classifier.fit(image_paths, labels)
    # predictions = classifier.predict(test_image_paths)
    print("Vision classifier initialized")


# Example 4: Multi-visualization (works for all modalities)
def multi_viz_example():
    """Example using multi-visualization framework."""
    classifier = ClamTsneClassifier(
        modality="tabular",  # or "audio" or "vision"
        vlm_model_id="Qwen/Qwen2-VL-2B-Instruct",
        enable_multi_viz=True,
        visualization_methods=["tsne", "pca", "umap"],
        layout_strategy="adaptive_grid",
        reasoning_focus="comparison"
    )
    
    # The classifier will now use multiple visualization methods
    # and compose them into a unified view for VLM reasoning
    print("Multi-visualization classifier initialized")


if __name__ == "__main__":
    print("=== Unified CLAM Examples ===\n")
    
    print("1. Tabular Example:")
    tabular_example()
    print()
    
    print("2. Audio Example:")
    audio_example()
    print()
    
    print("3. Vision Example:")
    vision_example()
    print()
    
    print("4. Multi-Visualization Example:")
    multi_viz_example()
    print()
    
    print("All classifiers are now available from clam.models!")