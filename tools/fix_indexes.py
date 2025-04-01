#!/usr/bin/env python
"""
Script to fix inconsistent indexing in image filenames
This script finds image groups where some have indices and some don't,
and ensures all images in a group have proper sequential indices
"""

import os
import re
import argparse
from collections import defaultdict
from tqdm import tqdm

def fix_indexes(directory, dry_run=False):
    """
    Find and fix inconsistently indexed image files in a directory
    
    Args:
        directory: Directory containing the image files
        dry_run: If True, only show what would be done without actually renaming
    """
    print(f"Scanning files in {directory} for inconsistent indexing...")
    
    # Regex to extract the base part (e.g., "0010_p_3") and optional index suffix
    image_pattern = re.compile(r'^(\d{4}_p_\d+)(?:_(\d+))?\.jpg$')
    
    # Group files by their base pattern
    file_groups = defaultdict(list)
    
    for filename in os.listdir(directory):
        if not filename.lower().endswith('.jpg'):
            continue
            
        match = image_pattern.match(filename)
        if not match:
            continue
            
        base_name = match.group(1)
        index = int(match.group(2)) if match.group(2) else None
        path = os.path.join(directory, filename)
        
        file_groups[base_name].append((path, index, filename))
    
    # Find groups with inconsistent indexing (mix of indexed and non-indexed files)
    problematic_groups = {}
    
    for base_name, files in file_groups.items():
        # Skip if there's only one file
        if len(files) <= 1:
            continue
            
        # Check if we have a mix of indexed and non-indexed files
        has_indexed = any(index is not None for _, index, _ in files)
        has_non_indexed = any(index is None for _, index, _ in files)
        
        if has_indexed or has_non_indexed:
            problematic_groups[base_name] = files
    
    if not problematic_groups:
        print(f"No inconsistently indexed files found in {directory}")
        return
    
    print(f"Found {len(problematic_groups)} groups with inconsistent indexing:")
    
    # Print the problematic groups
    for base_name, files in problematic_groups.items():
        filenames = [f for _, _, f in files]
        print(f"  {base_name}: {', '.join(filenames)}")
    
    # Fix the inconsistent groups
    renamed_count = 0
    
    for base_name, files in tqdm(problematic_groups.items(), desc="Fixing groups"):
        # Sort files - non-indexed first, then by index
        files.sort(key=lambda x: float('-inf') if x[1] is None else x[1])
        
        # Rename files with new sequential indices
        for i, (path, _, old_filename) in enumerate(files):
            new_filename = f"{base_name}_{i}.jpg"
            new_path = os.path.join(os.path.dirname(path), new_filename)
            
            if os.path.normpath(path) == os.path.normpath(new_path):
                continue
                
            if dry_run:
                print(f"Would rename: {old_filename} -> {new_filename}")
            else:
                try:
                    os.rename(path, new_path)
                    renamed_count += 1
                except Exception as e:
                    print(f"Error renaming {path}: {e}")
    
    print(f"Fixed {renamed_count} files with inconsistent indexing")
    if dry_run:
        print("This was a dry run, no files were actually renamed.")

def main():
    parser = argparse.ArgumentParser(description="Fix inconsistent indexing in image filenames.")
    parser.add_argument('directory', help='Directory containing the image files')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be done without actually renaming')
    
    args = parser.parse_args()
    
    fix_indexes(args.directory, args.dry_run)

if __name__ == "__main__":
    main()