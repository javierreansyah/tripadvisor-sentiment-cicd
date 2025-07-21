#!/usr/bin/env python3
"""
Admin DVC Versioning Automation Script

This script automates the DVC versioning process for the tripadvisor sentiment analysis project.
Run this script after training completion to version the updated data.

Usage:
    python admin_dvc_version.py [--message "Custom commit message"]

Requirements:
    - DVC must be installed and configured
    - Git repository must be properly set up
    - Run from the root directory of the project
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path


def run_command(cmd, check=True):
    """Execute a shell command and return the result."""
    print(f"Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr.strip()}")
        return None


def check_prerequisites():
    """Check if DVC and Git are available and we're in the right directory."""
    print("Checking prerequisites...")
    
    # Check if we're in the project root
    if not Path('.dvc').exists():
        print("ERROR: .dvc directory not found. Make sure you're running this script from the project root.")
        return False
    
    if not Path('.git').exists():
        print("ERROR: .git directory not found. Make sure you're in a git repository.")
        return False
    
    if not Path('Data/data.csv').exists():
        print("ERROR: Data/data.csv not found. Make sure the data file exists.")
        return False
    
    # Check if DVC is installed
    result = run_command(['dvc', '--version'], check=False)
    if result is None or result.returncode != 0:
        print("ERROR: DVC is not installed or not in PATH.")
        return False
    
    # Check if Git is installed
    result = run_command(['git', '--version'], check=False)
    if result is None or result.returncode != 0:
        print("ERROR: Git is not installed or not in PATH.")
        return False
    
    print("✓ All prerequisites met.")
    return True


def get_data_info():
    """Get information about the current data file."""
    data_path = Path('Data/data.csv')
    if data_path.exists():
        # Get file size
        size_mb = data_path.stat().st_size / (1024 * 1024)
        
        # Get number of lines (approximate number of records)
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
            return f"Size: {size_mb:.2f}MB, Records: ~{lines-1} (excluding header)"
        except Exception as e:
            return f"Size: {size_mb:.2f}MB, Could not count records: {e}"
    return "Data file not found"


def check_git_status():
    """Check if there are uncommitted changes."""
    result = run_command(['git', 'status', '--porcelain'], check=False)
    if result and result.stdout.strip():
        print("WARNING: There are uncommitted changes in the repository:")
        print(result.stdout)
        response = input("Do you want to continue? (y/N): ")
        if response.lower() != 'y':
            return False
    return True


def version_data_with_dvc(commit_message):
    """Perform the DVC versioning process."""
    print("\n" + "="*50)
    print("Starting DVC versioning process...")
    print("="*50)
    
    # Step 1: Add data to DVC
    print("\nStep 1: Adding data to DVC...")
    result = run_command(['dvc', 'add', 'Data/data.csv'])
    if result is None:
        return False
    
    # Step 2: Add DVC files to git
    print("\nStep 2: Adding DVC files to git...")
    result = run_command(['git', 'add', 'Data/data.csv.dvc', 'Data/.gitignore'])
    if result is None:
        return False
    
    # Step 3: Commit changes
    print("\nStep 3: Committing changes...")
    result = run_command(['git', 'commit', '-m', commit_message])
    if result is None:
        return False
    
    # Step 4: Ask about pushing
    print("\nStep 4: Push changes to remote repositories?")
    print("This will:")
    print("  - Push git changes to remote repository")
    print("  - Push DVC data to remote storage")
    
    response = input("Do you want to push now? (y/N): ")
    if response.lower() == 'y':
        print("\nPushing git changes...")
        git_result = run_command(['git', 'push'], check=False)
        
        print("\nPushing DVC data...")
        dvc_result = run_command(['dvc', 'push'], check=False)
        
        if git_result and git_result.returncode == 0 and dvc_result and dvc_result.returncode == 0:
            print("✓ Successfully pushed all changes!")
        else:
            print("⚠ Some push operations may have failed. Please check manually.")
    else:
        print("Skipping push. Remember to run 'git push' and 'dvc push' manually later.")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Automate DVC versioning for training data')
    parser.add_argument('--message', '-m', 
                       help='Custom commit message',
                       default=None)
    parser.add_argument('--auto-message', 
                       action='store_true',
                       help='Generate automatic commit message with timestamp')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TripAdvisor Sentiment Analysis - DVC Versioning Tool")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Show current data info
    print(f"\nCurrent data info: {get_data_info()}")
    
    # Check git status
    if not check_git_status():
        print("Aborted by user.")
        sys.exit(1)
    
    # Determine commit message
    if args.message:
        commit_message = args.message
    elif args.auto_message:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Data version update - {timestamp}"
    else:
        default_message = f"Data version update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        print(f"\nDefault commit message: '{default_message}'")
        user_message = input("Enter custom commit message (or press Enter for default): ").strip()
        commit_message = user_message if user_message else default_message
    
    print(f"\nCommit message: '{commit_message}'")
    
    # Confirm before proceeding
    response = input("\nProceed with DVC versioning? (y/N): ")
    if response.lower() != 'y':
        print("Aborted by user.")
        sys.exit(0)
    
    # Perform DVC versioning
    if version_data_with_dvc(commit_message):
        print("\n" + "="*50)
        print("✓ DVC versioning completed successfully!")
        print("="*50)
        print("\nNext steps:")
        print("1. Verify the changes in your git log")
        print("2. If you didn't push, remember to run:")
        print("   git push")
        print("   dvc push")
    else:
        print("\n" + "="*50)
        print("✗ DVC versioning failed!")
        print("="*50)
        print("Please check the error messages above and fix any issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
