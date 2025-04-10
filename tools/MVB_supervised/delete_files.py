#!/usr/bin/env python
"""
Delete files with IDs ranging from 4020 to 4518 in the MVB bounding_box_train dataset.
This script performs a dry run by default to show what files would be deleted.

Created: April 10, 2025
"""

import os
import argparse
import re
from pathlib import Path


def get_file_id(filename):
    """Extract ID from filename, e.g., '4020_g_1.jpg' -> 4020"""
    match = re.match(r'^(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None


def find_files_to_delete(directory, start_id=4020, end_id=4518):
    """Find files with IDs in the specified range."""
    files_to_delete = []
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return files_to_delete
    
    for filename in os.listdir(directory):
        file_id = get_file_id(filename)
        if file_id is not None and start_id <= file_id <= end_id:
            files_to_delete.append(os.path.join(directory, filename))
    
    return files_to_delete


def delete_files(files_list, dry_run=True):
    """Delete files in the list, or just print what would be deleted if dry_run is True."""
    if not files_list:
        print("No files found to delete.")
        return
    
    # Group files by ID for better summary
    files_by_id = {}
    for file_path in files_list:
        filename = os.path.basename(file_path)
        file_id = get_file_id(filename)
        if file_id not in files_by_id:
            files_by_id[file_id] = []
        files_by_id[file_id].append(filename)
    
    # Print summary
    print(f"Found {len(files_list)} files to delete across {len(files_by_id)} IDs")
    
    if dry_run:
        print("DRY RUN - No files will be deleted")
        # Print sample of files that would be deleted (first 5 IDs)
        sample_ids = list(files_by_id.keys())[:5]
        for file_id in sample_ids:
            print(f"ID {file_id}: {len(files_by_id[file_id])} files (e.g., {files_by_id[file_id][0]})")
        if len(files_by_id) > 5:
            print(f"... and {len(files_by_id) - 5} more IDs")
    else:
        print("Deleting files...")
        for file_path in files_list:
            try:
                os.remove(file_path)
                print(f"Deleted: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        
        print(f"Successfully deleted {len(files_list)} files.")


def main():
    parser = argparse.ArgumentParser(
        description="Delete files with IDs ranging from 4020 to 4518 from MVB dataset."
    )
    parser.add_argument(
        "--directory", 
        type=str,
        default="/data1/zhaofanghan/SuitcaseReID/OpenUnReID/datasets/MVB/bounding_box_train/", 
        help="Directory containing the files to delete"
    )
    parser.add_argument(
        "--start-id", 
        type=int, 
        default=4020, 
        help="Starting ID for deletion range"
    )
    parser.add_argument(
        "--end-id", 
        type=int, 
        default=4518, 
        help="Ending ID for deletion range"
    )
    parser.add_argument(
        "--execute", 
        action="store_true", 
        help="Execute actual deletion (without this flag, only a dry run is performed)"
    )
    
    args = parser.parse_args()
    
    print(f"Searching for files with IDs from {args.start_id} to {args.end_id} in {args.directory}")
    
    files_to_delete = find_files_to_delete(args.directory, args.start_id, args.end_id)
    
    delete_files(files_to_delete, dry_run=not args.execute)
    
    if not args.execute:
        print("\nThis was a dry run. To actually delete the files, run with the --execute flag.")


if __name__ == "__main__":
    main()
