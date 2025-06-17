#!/usr/bin/env python
"""
Script to check which files will be included in git based on .gitignore rules.
"""

import os
import subprocess
from pathlib import Path

def main():
    # Current directory
    current_dir = Path.cwd()
    print(f"Checking git status in: {current_dir}")
    
    # Run git status to see if git is initialized
    try:
        subprocess.run(["git", "status"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ Current directory is not a git repository. Initializing git...")
        subprocess.run(["git", "init"], check=True)
    
    # Check which files would be included in git with current .gitignore rules
    print("\nFiles that would be added to git (not ignored):")
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        check=True, capture_output=True, text=True
    )
    
    files = result.stdout.strip().split("\n")
    
    # Print the files with focus on llata/data directory
    if not files or (len(files) == 1 and not files[0]):
        print("  No untracked files found.")
    else:
        for file in files:
            if file.startswith("llata/data/"):
                print(f"  ✅ {file}")
            else:
                print(f"  {file}")
    
    # Also check tracked files
    print("\nAlready tracked files:")
    result = subprocess.run(
        ["git", "ls-files"],
        check=True, capture_output=True, text=True
    )
    
    files = result.stdout.strip().split("\n")
    
    # Print the files with focus on llata/data directory
    if not files or (len(files) == 1 and not files[0]):
        print("  No tracked files found.")
    else:
        for file in files:
            if file.startswith("llata/data/"):
                print(f"  ✅ {file}")
            else:
                print(f"  {file}")
    
    # Check specifically for llata/data directory
    print("\nChecking llata/data directory:")
    llata_data_dir = current_dir / "llata" / "data"
    if not llata_data_dir.exists():
        print(f"  ❌ Directory {llata_data_dir} does not exist!")
        return
    
    print(f"  ✅ Directory {llata_data_dir} exists")
    
    # List all files in llata/data directory
    print("\nFiles in llata/data directory:")
    for file in llata_data_dir.rglob("*"):
        if file.is_file():
            rel_path = file.relative_to(current_dir)
            ignored = is_ignored(rel_path)
            if ignored:
                print(f"  ❌ {rel_path} (would be ignored by git)")
            else:
                print(f"  ✅ {rel_path} (would be included in git)")

def is_ignored(file_path):
    """Check if a file would be ignored by git."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(file_path)],
        capture_output=True
    )
    return result.returncode == 0

if __name__ == "__main__":
    main()