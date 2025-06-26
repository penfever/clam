#!/usr/bin/env python3
"""
Test script to verify Decision-SVC fix works without class_names error.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification

def test_decision_svc():
    """Test Decision-SVC plotting with class_names parameter."""
    print("Testing Decision-SVC fix...")
    
    try:
        from clam.viz.decision.regions import create_decision_regions_visualization
        
        # Create synthetic 2D data for decision regions
        X, y = make_classification(
            n_samples=100, 
            n_features=2, 
            n_redundant=0, 
            n_informative=2,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Create and train classifier
        classifier = SVC(kernel='rbf', random_state=42)
        
        # Test with class_names parameter (this should not cause an error)
        result = create_decision_regions_visualization(
            X_train=X,
            y_train=y,
            classifier=classifier,
            embedding_method='pca',
            random_state=42,
            class_names=['Class A', 'Class B'],  # This used to cause an error
            use_semantic_names=True  # This too
        )
        
        # Save the result
        viz_result = result['visualization_result']
        viz_result.image.save('/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/clam/test_decision_svc.png')
        
        print("✓ Decision-SVC test completed successfully")
        print(f"  - Classifier: {result['classifier'].__class__.__name__}")
        print(f"  - Train embedding shape: {result['train_embedding'].shape}")
        print(f"  - Visualization method: {viz_result.method_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Decision-SVC test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Decision-SVC fix...")
    
    success = test_decision_svc()
    
    if success:
        print("✓ Decision-SVC fix is working correctly!")
    else:
        print("✗ Decision-SVC fix needs more work.")