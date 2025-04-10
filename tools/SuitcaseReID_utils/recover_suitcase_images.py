#!/usr/bin/env python
"""
Script to recover suitcase images from the original dataset
and recreate the SuitcaseReID_Multiview dataset with proper indexing
"""

import os
import re
import shutil
import argparse
from collections import defaultdict
from tqdm import tqdm
import time

def extract_suitcase_reid(source_dirs, target_dir, pattern=r'(\d{4})_p_(\d+)', dry_run=False):
    """
    Extract images with a specific pattern to a new dataset directory
    
    Args:
        source_dirs: List of source directories to search
        target_dir: Target directory to copy files to
        pattern: Regex pattern to match filenames
        dry_run: If True, only show what would be done without actually copying
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Timestamp for creating backup
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = f"{target_dir}_backup_{timestamp}"
    
    # Create a backup of the current target directory if it exists and has files
    if os.path.exists(target_dir) and os.listdir(target_dir) and not dry_run:
        print(f"Creating backup of current files at {backup_dir}")
        shutil.copytree(target_dir, backup_dir)
        print(f"Backup created at {backup_dir}")
    
    # Regular expression for matching the specified pattern
    regex = re.compile(f"{pattern}.*\.jpg$")
    
    # Find all matching files in source directories
    matching_files = []
    print(f"Scanning source directories for pattern '{pattern}'...")
    
    for source_dir in source_dirs:
        print(f"Searching in {source_dir}...")
        for root, _, files in os.walk(source_dir):
            for filename in files:
                if regex.match(filename):
                    source_path = os.path.join(root, filename)
                    matching_files.append((source_path, filename))
    
    print(f"Found {len(matching_files)} files matching pattern '{pattern}'")
    
    # Group files by their base pattern (e.g., "0010_p_3")
    file_groups = defaultdict(list)
    
    for source_path, filename in matching_files:
        # Extract base pattern (e.g., "0010_p_3")
        base_match = re.match(f"({pattern}).*\.jpg$", filename)
        if base_match:
            base_pattern = base_match.group(1)
            file_groups[base_pattern].append((source_path, filename))
    
    print(f"Found {len(file_groups)} unique groups of files")
    
    # Process each group of files
    copied_count = 0
    
    for base_pattern, files in tqdm(file_groups.items(), desc="Processing groups"):
        # Sort files for deterministic indexing
        files.sort(key=lambda x: x[1])
        
        # Assign new indices
        for i, (source_path, _) in enumerate(files):
            new_filename = f"{base_pattern}_{i}.jpg"
            target_path = os.path.join(target_dir, new_filename)
            
            if dry_run:
                print(f"Would copy: {source_path} -> {target_path}")
            else:
                try:
                    shutil.copy2(source_path, target_path)
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying {source_path}: {e}")
    
    print(f"{'Would copy' if dry_run else 'Copied'} {copied_count} files to {target_dir}")
    if dry_run:
        print("This was a dry run, no files were actually copied.")
    else:
        print(f"Recovery complete. Backup saved at {backup_dir}")

def main():
    parser = argparse.ArgumentParser(description="Recover suitcase images from original dataset.")
    parser.add_argument('--source-dirs', nargs='+', required=True,
                        help='Source directories to search for images')
    parser.add_argument('--target-dir', required=True,
                        help='Target directory to copy files to')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without actually copying')
    
    args = parser.parse_args()
    
    extract_suitcase_reid(args.source_dirs, args.target_dir, dry_run=args.dry_run)

if __name__ == "__main__":
    main()