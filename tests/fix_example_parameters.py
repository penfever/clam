#!/usr/bin/env python
"""
Script to automatically fix deprecated parameter usage in CLAM examples.

This script will:
1. Replace use_3d_tsne with use_3d
2. Replace --use_3d_tsne with --use_3d
3. Update getattr calls for deprecated parameters
4. Create backups of modified files
"""

import os
import re
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Parameter replacements
PARAMETER_REPLACEMENTS = {
    'use_3d_tsne': 'use_3d',
    'tsne_zoom_factor': 'zoom_factor',
}

class ParameterFixer:
    def __init__(self, examples_dir: str, create_backups: bool = True):
        self.examples_dir = Path(examples_dir)
        self.create_backups = create_backups
        self.files_modified = 0
        self.total_fixes = 0
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the examples directory."""
        python_files = []
        for root, dirs, files in os.walk(self.examples_dir):
            # Skip __pycache__ and .git directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    python_files.append(Path(root) / file)
        return python_files
    
    def fix_file(self, file_path: Path) -> int:
        """Fix parameter issues in a single file. Returns number of fixes made."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            fixes_made = 0
            
            # Apply all fixes
            for old_param, new_param in PARAMETER_REPLACEMENTS.items():
                # Fix parameter assignments
                patterns_and_replacements = [
                    # Parameter assignment with =
                    (rf'{old_param}=', f'{new_param}='),
                    (rf'{old_param} =', f'{new_param} ='),
                    
                    # Command line arguments
                    (rf'--{old_param}', f'--{new_param}'),
                    
                    # getattr calls
                    (rf"getattr\(args, '{old_param}'", f"getattr(args, '{new_param}'"),
                    (rf'getattr\(args, "{old_param}"', f'getattr(args, "{new_param}"'),
                    
                    # Class attribute assignments
                    (rf'self\.{old_param} =', f'self.{new_param} ='),
                ]
                
                for pattern, replacement in patterns_and_replacements:
                    new_content, count = re.subn(pattern, replacement, content)
                    if count > 0:
                        content = new_content
                        fixes_made += count
                        print(f"  Fixed {count} instances of '{pattern}' -> '{replacement}'")
            
            # If we made any changes, write the file
            if fixes_made > 0:
                # Create backup if requested
                if self.create_backups:
                    backup_path = file_path.with_suffix('.py.bak')
                    shutil.copy2(file_path, backup_path)
                    print(f"  Created backup: {backup_path}")
                
                # Write the fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.files_modified += 1
                self.total_fixes += fixes_made
                print(f"  ✅ Fixed {fixes_made} issues in {file_path}")
            
            return fixes_made
            
        except Exception as e:
            print(f"  ❌ Error fixing file {file_path}: {e}")
            return 0
    
    def run_fixes(self) -> bool:
        """Run fixes on all example files."""
        print(f"🔧 Fixing parameter usage in examples directory: {self.examples_dir}")
        print("=" * 80)
        
        python_files = self.find_python_files()
        
        if not python_files:
            print("❌ No Python files found in examples directory")
            return False
        
        print(f"Found {len(python_files)} Python files to check...")
        print()
        
        for file_path in python_files:
            print(f"Checking: {file_path}")
            fixes_made = self.fix_file(file_path)
            if fixes_made == 0:
                print(f"  ✓ No issues found")
            print()
        
        return self._report_results()
    
    def _report_results(self) -> bool:
        """Report fix results."""
        print("=" * 80)
        print("🎯 PARAMETER FIX RESULTS")
        print("=" * 80)
        
        if self.total_fixes == 0:
            print(f"✅ No fixes needed - all examples already use correct parameter names!")
        else:
            print(f"✅ SUCCESS: Fixed {self.total_fixes} parameter issues in {self.files_modified} files!")
        
        print()
        print("📋 Fix Summary:")
        print(f"  • Files checked: {len(self.find_python_files())}")
        print(f"  • Files modified: {self.files_modified}")
        print(f"  • Total fixes applied: {self.total_fixes}")
        print(f"  • Backups created: {'Yes' if self.create_backups else 'No'}")
        
        if self.total_fixes > 0:
            print()
            print("🔄 Next steps:")
            print("  1. Review the changes in the modified files")
            print("  2. Test that the examples still work correctly")
            print("  3. Run the validation test again to confirm all issues are fixed")
            if self.create_backups:
                print("  4. Remove backup files (.py.bak) when satisfied with changes")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Fix deprecated parameter usage in CLAM examples")
    parser.add_argument("--examples-dir", default="examples", 
                        help="Path to examples directory (default: examples)")
    parser.add_argument("--no-backup", action="store_true",
                        help="Don't create backup files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fixed without making changes")
    
    args = parser.parse_args()
    
    # Determine examples directory path
    examples_dir = Path(args.examples_dir)
    if not examples_dir.is_absolute():
        examples_dir = Path(__file__).parent.parent / examples_dir
    
    if not examples_dir.exists():
        print(f"❌ Examples directory not found: {examples_dir}")
        return 1
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print()
    
    # Run fixes
    fixer = ParameterFixer(str(examples_dir), create_backups=not args.no_backup)
    
    if args.dry_run:
        # For dry run, just show what would be fixed
        print("This would fix the following parameter issues:")
        # Import the validator to show current issues
        sys.path.append(str(Path(__file__).parent))
        from test_example_parameter_validation import ExampleParameterValidator
        validator = ExampleParameterValidator(str(examples_dir))
        validator.run_validation()
        return 0
    else:
        success = fixer.run_fixes()
        return 0 if success else 1

if __name__ == "__main__":
    exit(main())