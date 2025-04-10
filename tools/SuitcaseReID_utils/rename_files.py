#!/usr/bin/env python
"""
Script to rename image files by removing the .rf.[hash] part from filenames
and handle duplicate prefixes by adding indices
"""

import os
import re
import argparse
import shutil
from tqdm import tqdm
from collections import defaultdict

def rename_files_with_index(directory, dry_run=False):
    """
    Rename files in the directory by removing the .rf.[hash] part from filenames
    and handle duplicates by adding indices
    
    Args:
        directory: Directory containing the image files
        dry_run: If True, only show what would be done without actually renaming
    """
    # Regular expression to match the .rf.[hash] part
    pattern = r'(.+?)_jpg\.rf\.[a-z0-9]+\.jpg$'
    
    renamed_count = 0
    
    print(f"Scanning for files to rename in {directory}...")
    
    # First pass: group files by their would-be base name after removing .rf.[hash]
    prefix_groups = defaultdict(list)
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.jpg'):
                match = re.match(pattern, filename)
                if match:
                    # Get the base name without the .rf.[hash] part
                    base_prefix = match.group(1)
                    old_path = os.path.join(root, filename)
                    prefix_groups[os.path.join(root, base_prefix)].append(old_path)
    
    # Second pass: rename files, adding indices for duplicates
    for base_prefix, file_paths in tqdm(prefix_groups.items(), desc="Renaming files"):
        for i, old_path in enumerate(sorted(file_paths)):
            # For single files, don't add an index; for duplicates, add _0, _1, etc.
            suffix = "" if len(file_paths) == 1 else f"_{i}"
            new_path = f"{base_prefix}{suffix}.jpg"
            
            if old_path == new_path:
                continue
                
            if dry_run:
                print(f"Would rename: {old_path} -> {new_path}")
            else:
                try:
                    os.rename(old_path, new_path)
                    renamed_count += 1
                except Exception as e:
                    print(f"Error renaming {old_path}: {e}")
    
    print(f"Renamed {renamed_count} files")
    if dry_run:
        print("This was a dry run, no files were actually renamed.")
    
    return renamed_count

def extract_suitcase_reid(source_dirs, target_dir, pattern=r'(\d{4})_p_(\d+)', dry_run=False):
    """
    Extract images with a specific pattern to a new dataset directory
    
    Args:
        source_dirs: List of source directories to search
        target_dir: Target directory to move files to
        pattern: Regex pattern to match filenames
        dry_run: If True, only show what would be done without actually moving
    """
    moved_count = 0
    regex = re.compile(f"{pattern}.*\.jpg$")
    
    if not dry_run and not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    print(f"Scanning for SuitcaseReID pattern files to move...")
    
    for source_dir in source_dirs:
        for root, _, files in os.walk(source_dir):
            for filename in files:
                if regex.match(filename):
                    source_path = os.path.join(root, filename)
                    target_path = os.path.join(target_dir, filename)
                    
                    if dry_run:
                        print(f"Would move: {source_path} -> {target_path}")
                    else:
                        try:
                            shutil.move(source_path, target_path)
                            moved_count += 1
                        except Exception as e:
                            print(f"Error moving {source_path}: {e}")
    
    print(f"Moved {moved_count} files to {target_dir}")
    if dry_run:
        print("This was a dry run, no files were actually moved.")
    
    return moved_count

def main():
    parser = argparse.ArgumentParser(description="Rename and organize image files.")
    parser.add_argument('--rename-dir', help='Directory containing the image files to rename')
    parser.add_argument('--extract-suitcase', action='store_true',
                        help='Extract SuitcaseReID pattern images to a new dataset')
    parser.add_argument('--source-dirs', nargs='+', 
                        help='Source directories for SuitcaseReID extraction')
    parser.add_argument('--target-dir', default='datasets/SuitcaseReID_Multiview',
                        help='Target directory for SuitcaseReID extraction')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be done without actually renaming/moving')
    
    args = parser.parse_args()
    
    if args.rename_dir:
        rename_files_with_index(args.rename_dir, args.dry_run)
    
    if args.extract_suitcase:
        if not args.source_dirs:
            print("Error: --source-dirs must be provided for SuitcaseReID extraction")
            return
        extract_suitcase_reid(args.source_dirs, args.target_dir, dry_run=args.dry_run)

if __name__ == "__main__":
    main()