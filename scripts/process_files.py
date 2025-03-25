"""
Directory Processing and Cleanup Script

This script processes directories by:
1. Cleaning up non-PNG files (deleting files that don't end in .png)
2. Renaming PNG files using a three-digit numerical system (001, 002, etc.)
3. Optionally sorting files before renaming
4. Optionally processing subdirectories recursively

Note: This script only processes files, not directories. When using the --recursive flag,
      all directory names and structure remain unchanged. Only files within directories
      are processed.

Usage:
    python3.9 process_files.py --directory <path> [--sort] [--recursive]

Arguments:
    --directory: Path to the directory to process
    --sort: Optional flag to sort files before renaming
    --recursive: Optional flag to process subdirectories recursively

Example:
    # Process current directory only
    python3.9 process_files.py --directory ./data/images

    # Sort and process current directory
    python3.9 process_files.py --directory ./data/images --sort

    # Process recursively in all subdirectories
    python3.9 process_files.py --directory ./data/images --recursive

    # Sort and process recursively in all subdirectories
    python3.9 process_files.py --directory ./data/images --sort --recursive

Example Directory Structure:
    Before:
    data/
    ├── images/
    │   ├── subdir1/
    │   │   ├── file1.png
    │   │   ├── file2.jpg
    │   │   └── .DS_Store
    │   └── subdir2/
    │       ├── file3.png
    │       └── file4.txt

    After (with --recursive):
    data/
    ├── images/
    │   ├── subdir1/
    │   │   ├── 001.png
    │   │   └── 002.png
    │   └── subdir2/
    │       └── 001.png
"""

import os
import argparse
from pathlib import Path
import shutil
from typing import List, Tuple

def get_all_files(directory: str) -> List[Path]:
    """
    Get all files in the specified directory.
    
    Args:
        directory (str): Path to the directory
        
    Returns:
        List[Path]: List of file paths
    """
    return [f for f in Path(directory).iterdir() if f.is_file()]

def get_png_files(directory: str) -> List[Path]:
    """
    Get all PNG files in the specified directory.
    Excludes .DS_Store files which are macOS system files.
    
    Args:
        directory (str): Path to the directory
        
    Returns:
        List[Path]: List of PNG file paths
    """
    return [f for f in Path(directory).iterdir() 
            if f.is_file() 
            and f.name != ".DS_Store"
            and f.suffix.lower() == ".png"]

def get_directories(directory: str) -> List[Path]:
    """
    Get all subdirectories in the specified directory.
    
    Args:
        directory (str): Path to the directory
        
    Returns:
        List[Path]: List of directory paths
    """
    return [d for d in Path(directory).iterdir() if d.is_dir()]

def sort_files(files: List[Path]) -> List[Path]:
    """
    Sort files by name.
    
    Args:
        files (List[Path]): List of file paths
        
    Returns:
        List[Path]: Sorted list of file paths
    """
    return sorted(files)

def cleanup_directory(directory: str) -> None:
    """
    Delete all non-PNG files in the directory.
    
    Args:
        directory (str): Path to the directory
    """
    all_files = get_all_files(directory)
    png_files = get_png_files(directory)
    
    # Find files to delete (all files except PNG files)
    files_to_delete = [f for f in all_files if f not in png_files]
    
    if files_to_delete:
        print(f"  Cleaning up {len(files_to_delete)} non-PNG files...")
        for file_path in files_to_delete:
            print(f"    Deleting: {file_path.name}")
            file_path.unlink()
        print("  Cleanup complete!")

def rename_files(files: List[Path], directory: str) -> None:
    """
    Rename files using three-digit numerical system.
    Only renames existing files, never creates new ones.
    
    Args:
        files (List[Path]): List of file paths to rename
        directory (str): Path to the directory containing the files
    """
    # Get list of existing files to ensure we only rename what exists
    existing_files = [f for f in files if f.exists()]
    
    # Rename only existing files
    for idx, file_path in enumerate(existing_files, 1):
        if not file_path.exists():
            print(f"  Warning: File {file_path} no longer exists, skipping...")
            continue
            
        new_name = f"{idx:03d}{file_path.suffix}"
        new_path = Path(directory) / new_name
        
        # Only proceed if the new name doesn't already exist
        if new_path.exists():
            print(f"  Warning: File {new_name} already exists, skipping...")
            continue
            
        print(f"  Renaming {file_path.name} to {new_name}")
        file_path.rename(new_path)

def process_directory(directory: str, sort_files_flag: bool, recursive: bool) -> None:
    """
    Process a directory by cleaning up non-PNG files and renaming PNG files.
    
    Args:
        directory (str): Path to the directory to process
        sort_files_flag (bool): Whether to sort files before renaming
        recursive (bool): Whether to process subdirectories recursively
    """
    print(f"\nProcessing directory: {directory}")
    
    # First, clean up non-PNG files
    cleanup_directory(directory)
    
    # Then get remaining PNG files
    files = get_png_files(directory)
    
    if files:
        print(f"Found {len(files)} PNG files")
        
        # Sort files if requested
        if sort_files_flag:
            print("Sorting files...")
            files = sort_files(files)
        
        # Rename files
        print("Renaming files...")
        rename_files(files, directory)
        print("Done!")
    
    # Process subdirectories if recursive flag is set
    if recursive:
        subdirs = get_directories(directory)
        for subdir in subdirs:
            print(f"\nEntering subdirectory: {subdir}")
            process_directory(str(subdir), sort_files_flag, recursive)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process directories by cleaning up non-PNG files and renaming PNG files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--directory", required=True, help="Directory to process")
    parser.add_argument("--sort", action="store_true", help="Sort files before renaming")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively")
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        return
    
    # Process the directory
    process_directory(args.directory, args.sort, args.recursive)
    print("\nAll done!")

if __name__ == "__main__":
    main() 