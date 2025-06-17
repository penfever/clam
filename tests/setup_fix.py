#!/usr/bin/env python
"""
Script to fix any issues with the clam package installation.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def main():
    # Get the current directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we're in the clam directory
    if current_dir.name != "clam":
        print("❌ This script should be run from the clam directory.")
        sys.exit(1)
    
    # Make sure all __init__.py files exist
    print("Ensuring all necessary __init__.py files exist...")
    
    directories = [
        "clam",
        "clam/data",
        "clam/models",
        "clam/train",
        "clam/utils"
    ]
    
    for dir_path in directories:
        init_file = current_dir / dir_path / "__init__.py"
        dir_folder = current_dir / dir_path
        
        # Create directory if it doesn't exist
        if not dir_folder.exists():
            print(f"  Creating directory {dir_folder}")
            dir_folder.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py if it doesn't exist
        if not init_file.exists():
            print(f"  Creating {init_file}")
            with open(init_file, "w") as f:
                f.write('"""Auto-generated __init__.py file."""\n')
    
    # Update the main __init__.py file
    main_init = current_dir / "clam" / "__init__.py"
    print(f"Updating {main_init}")
    with open(main_init, "w") as f:
        f.write('''"""
CLAM: Classify Anything Model
A library for fine-tuning LLMs on tabular data using embeddings from tabular foundation models.
"""

__version__ = "0.1.0"

# Import main modules
from . import data
from . import models
from . import train
from . import utils

# Make submodules available directly
from .data import load_dataset, get_tabpfn_embeddings, create_llm_dataset
from .models import prepare_qwen_with_prefix_embedding
from .train import train_llm_with_tabpfn_embeddings, evaluate_llm_on_test_set
from .utils import setup_logging

__all__ = [
    'data',
    'models',
    'train',
    'utils',
    'load_dataset',
    'get_tabpfn_embeddings', 
    'create_llm_dataset',
    'prepare_qwen_with_prefix_embedding',
    'train_llm_with_tabpfn_embeddings',
    'evaluate_llm_on_test_set',
    'setup_logging'
]
''')
    
    # Uninstall existing clam package
    print("Uninstalling any existing clam package...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "clam"], check=False)
    
    # Install the package in development mode
    print("Installing clam package in development mode...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    
    # Run the test script
    print("\nRunning test script to verify installation...")
    subprocess.run([sys.executable, "test_install.py"], check=True)
    
    print("\n✅ Setup fix completed. If there are still issues, please check the error messages above.")

if __name__ == "__main__":
    main()