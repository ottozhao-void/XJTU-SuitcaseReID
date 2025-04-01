#!/usr/bin/env python
"""
Comprehensive script to restore and clean up the Airport Checked Bags dataset:
1. Extract the dataset from zip
2. Remove single-color images
3. Rename files properly (removing rf.hash)
4. Create a clean SuitcaseReID_Multiview dataset with consistent naming
"""

import os
import re
import shutil
import zipfile
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

def extract_dataset(zip_path, output_dir, overwrite=False):
    """
    Extract the dataset from a zip file
    """
    print(f"Extracting {zip_path} to {output_dir}...")
    
    if os.path.exists(output_dir) and overwrite:
        print(f"Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    print(f"Extraction complete. Dataset extracted to {output_dir}")

def is_single_color_image(img_path, std_threshold=2.0):
    """
    Check if an image has mostly uniform color (low standard deviation)
    """
    try:
        img = Image.open(img_path)
        data = np.array(img)
        
        # Calculate standard deviation across all pixels and channels
        std_value = data.std()
        
        # Check if standard deviation is very low (uniform color)
        return std_value <= std_threshold
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def clean_single_color_images(directory, std_threshold=2.0):
    """
    Remove single-color images from the dataset
    """
    print(f"Scanning for single-color images in {directory}...")
    
    removed_count = 0
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, filename)
                if is_single_color_image(img_path, std_threshold):
                    os.remove(img_path)
                    removed_count += 1
                    if removed_count % 50 == 0:
                        print(f"Removed {removed_count} single-color images so far...")
    
    print(f"Cleaning complete. Removed {removed_count} single-color images")

def rename_files(directory):
    """
    Rename files by removing the .rf.[hash] part from filenames
    """
    print(f"Renaming files in {directory} to remove .rf.[hash] pattern...")
    
    # Pattern to match: "filename_jpg.rf.hash.jpg"
    pattern = re.compile(r'(.+?)_jpg\.rf\.[a-z0-9]+\.jpg$')
    
    renamed_count = 0
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.jpg'):
                match = pattern.match(filename)
                if match:
                    base_name = match.group(1) + '.jpg'
                    old_path = os.path.join(root, filename)
                    new_path = os.path.join(root, base_name)
                    
                    # Handle duplicate base names by adding an index
                    if os.path.exists(new_path):
                        index = 0
                        while os.path.exists(os.path.join(root, f"{match.group(1)}_{index}.jpg")):
                            index += 1
                        new_path = os.path.join(root, f"{match.group(1)}_{index}.jpg")
                    
                    os.rename(old_path, new_path)
                    renamed_count += 1
    
    print(f"Renaming complete. Renamed {renamed_count} files")

def create_suitcase_reid_dataset(source_dirs, target_dir):
    """
    Create a clean SuitcaseReID_Multiview dataset from the source directories
    """
    print(f"Creating SuitcaseReID_Multiview dataset at {target_dir}...")
    
    # Clear existing target directory
    if os.path.exists(target_dir):
        print(f"Removing existing directory: {target_dir}")
        shutil.rmtree(target_dir)
    
    os.makedirs(target_dir)
    
    # Regular expression for matching the specified pattern
    pattern = re.compile(r'(\d{4})_p_(\d+).*\.jpg$')
    
    # Find and group all matching files
    file_groups = defaultdict(list)
    
    for source_dir in source_dirs:
        for root, _, files in os.walk(source_dir):
            for filename in files:
                match = pattern.match(filename)
                if match:
                    base_pattern = f"{match.group(1)}_p_{match.group(2)}"
                    source_path = os.path.join(root, filename)
                    file_groups[base_pattern].append(source_path)
    
    # Copy and rename files with consistent indexing
    copied_count = 0
    
    for base_pattern, file_paths in tqdm(file_groups.items(), desc="Processing"):
        for i, source_path in enumerate(sorted(file_paths)):
            target_path = os.path.join(target_dir, f"{base_pattern}_{i}.jpg")
            shutil.copy2(source_path, target_path)
            copied_count += 1
    
    print(f"Created SuitcaseReID_Multiview dataset with {copied_count} files")

def main():
    parser = argparse.ArgumentParser(description="Restore and clean up Airport Checked Bags dataset")
    parser.add_argument('--zip-path', default='/data1/zhaofanghan/OpenUnReID/datasets/Airport Checked Bags/Airport Checked Bags.v2i.yolov7pytorch.zip',
                        help='Path to the dataset zip file')
    parser.add_argument('--output-dir', default='/data1/zhaofanghan/OpenUnReID/datasets/Airport Checked Bags',
                        help='Output directory for the extracted dataset')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files')
    parser.add_argument('--std-threshold', type=float, default=2.0,
                        help='Standard deviation threshold for detecting single-color images')
    parser.add_argument('--reid-target-dir', default='/data1/zhaofanghan/OpenUnReID/datasets/SuitcaseReID_Multiview',
                        help='Target directory for SuitcaseReID_Multiview dataset')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip extracting the dataset (use if already extracted)')
    parser.add_argument('--skip-clean', action='store_true',
                        help='Skip cleaning single-color images')
    parser.add_argument('--skip-rename', action='store_true',
                        help='Skip renaming files')
    parser.add_argument('--skip-reid', action='store_true',
                        help='Skip creating SuitcaseReID_Multiview dataset')
    
    args = parser.parse_args()
    
    # Step 1: Extract the dataset
    if not args.skip_extract:
        extract_dataset(args.zip_path, args.output_dir, args.overwrite)
    
    # Step 2: Clean single-color images
    if not args.skip_clean:
        # Look for images in train, test, valid directories
        for subdir in ['train/images', 'test/images', 'valid/images']:
            dir_path = os.path.join(args.output_dir, subdir)
            if os.path.exists(dir_path):
                clean_single_color_images(dir_path, args.std_threshold)
    
    # Step 3: Rename files
    if not args.skip_rename:
        # Look for images in train, test, valid directories
        for subdir in ['train/images', 'test/images', 'valid/images']:
            dir_path = os.path.join(args.output_dir, subdir)
            if os.path.exists(dir_path):
                rename_files(dir_path)
    
    # Step 4: Create SuitcaseReID_Multiview dataset
    if not args.skip_reid:
        source_dirs = []
        for subdir in ['train/images', 'test/images', 'valid/images']:
            dir_path = os.path.join(args.output_dir, subdir)
            if os.path.exists(dir_path):
                source_dirs.append(dir_path)
        
        create_suitcase_reid_dataset(source_dirs, args.reid_target_dir)
    
    print("All operations completed successfully!")

if __name__ == "__main__":
    main()