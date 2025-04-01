#!/usr/bin/env python
"""
Script to reindex images with prefixes listed in affected_prefixes.txt
Ensures that all indices are sequential (0, 1, 2, ...) for each affected prefix
"""

import os
import re
import argparse
from collections import defaultdict
from tqdm import tqdm

def reindex_prefixes(directory, prefixes_file):
    """
    Reindex images with affected prefixes to ensure sequential indices
    
    Args:
        directory: Directory containing the images
        prefixes_file: File containing affected prefixes, one per line
    """
    print(f"Reading affected prefixes from {prefixes_file}")
    
    # Read the affected prefixes
    with open(prefixes_file, 'r') as f:
        affected_prefixes = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(affected_prefixes)} affected prefixes to reindex")
    
    # Group files by their prefix
    file_groups = defaultdict(list)
    
    for filename in os.listdir(directory):
        if not filename.lower().endswith('.jpg'):
            continue
            
        # Match the prefix pattern (e.g., "0010_p_3")
        match = re.match(r'^(\d{4}_p_\d+).*\.jpg$', filename)
        if match and match.group(1) in affected_prefixes:
            base_prefix = match.group(1)
            file_path = os.path.join(directory, filename)
            file_groups[base_prefix].append(file_path)
    
    # Reindex each group
    reindexed_count = 0
    
    for prefix, file_paths in tqdm(file_groups.items(), desc="Reindexing prefixes"):
        print(f"Reindexing {len(file_paths)} images with prefix {prefix}")
        
        # Sort files to ensure deterministic indexing
        sorted_paths = sorted(file_paths)
        
        # Reindex files
        for i, old_path in enumerate(sorted_paths):
            new_filename = f"{prefix}_{i}.jpg"
            new_path = os.path.join(os.path.dirname(old_path), new_filename)
            
            if old_path != new_path:
                os.rename(old_path, new_path)
                reindexed_count += 1
    
    print(f"Reindexing complete. Reindexed {reindexed_count} files across {len(affected_prefixes)} prefixes.")

def main():
    parser = argparse.ArgumentParser(description="Reindex images with affected prefixes")
    parser.add_argument('directory', help='Directory containing the images')
    parser.add_argument('--prefixes-file', default=None, 
                        help='File containing affected prefixes (default: affected_prefixes.txt in parent directory)')
    
    args = parser.parse_args()
    
    # Default prefixes file if not specified
    if args.prefixes_file is None:
        args.prefixes_file = os.path.join(os.path.dirname(args.directory), "affected_prefixes.txt")
    
    reindex_prefixes(args.directory, args.prefixes_file)

if __name__ == "__main__":
    main()