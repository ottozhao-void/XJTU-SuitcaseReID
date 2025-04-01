#!/usr/bin/env python
"""
Script to detect and remove uniform color images from a dataset.
This is useful for cleaning up datasets with corrupted, empty, or low-information images.
"""

import os
import argparse
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm

def is_uniform_color_image(img_path, std_threshold=2.0):
    """
    Check if an image has uniform color (all pixels very similar).
    
    Args:
        img_path: Path to the image
        std_threshold: Standard deviation threshold below which an image is 
                      considered uniform color (lower = stricter)
    
    Returns:
        True if the image is uniform color, False otherwise
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

def find_uniform_color_images(directory, std_threshold=2.0, 
                             recursive=True, extensions=('.jpg', '.jpeg', '.png')):
    """
    Find all uniform color images in a directory.
    
    Args:
        directory: Directory to search for images
        std_threshold: Standard deviation threshold
        recursive: Whether to search recursively
        extensions: Image file extensions to consider
    
    Returns:
        List of paths to uniform color images
    """
    uniform_images = []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(extensions):
                    img_path = os.path.join(root, file)
                    if is_uniform_color_image(img_path, std_threshold):
                        uniform_images.append(img_path)
    else:
        for file in os.listdir(directory):
            if file.lower().endswith(extensions):
                img_path = os.path.join(directory, file)
                if is_uniform_color_image(img_path, std_threshold):
                    uniform_images.append(img_path)
    
    return uniform_images

def process_dataset(directory, std_threshold=2.0, action='move', dest_dir=None, recursive=True):
    """
    Process a dataset by finding and handling uniform color images.
    
    Args:
        directory: Directory containing the dataset
        std_threshold: Standard deviation threshold
        action: Action to take for uniform color images ('move', 'delete', or 'list')
        dest_dir: Destination directory for moved images
        recursive: Whether to search recursively
    """
    print(f"Scanning for uniform color images in {directory}...")
    uniform_images = find_uniform_color_images(directory, std_threshold, recursive)
    
    print(f"Found {len(uniform_images)} uniform color images.")
    
    if not uniform_images:
        return
    
    if action == 'list':
        print("\nUniform color images:")
        for img_path in uniform_images:
            print(img_path)
        return
    
    if action == 'move':
        if not dest_dir:
            dest_dir = os.path.join(os.path.dirname(directory), 'uniform_color_images')
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        print(f"Moving {len(uniform_images)} uniform color images to {dest_dir}")
        for img_path in tqdm(uniform_images):
            # Preserve directory structure relative to original directory
            rel_path = os.path.relpath(img_path, directory)
            new_path = os.path.join(dest_dir, rel_path)
            
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            
            shutil.move(img_path, new_path)
    
    elif action == 'delete':
        print(f"Deleting {len(uniform_images)} uniform color images...")
        for img_path in tqdm(uniform_images):
            os.remove(img_path)

def main():
    parser = argparse.ArgumentParser(description="Detect and handle uniform color images in a dataset.")
    parser.add_argument('directory', help='Directory containing the image dataset')
    parser.add_argument('--std-threshold', type=float, default=2.0,
                        help='Standard deviation threshold (default: 2.0)')
    parser.add_argument('--action', choices=['move', 'delete', 'list'], default='list',
                        help='Action to take for uniform color images (default: list)')
    parser.add_argument('--dest-dir', help='Destination directory for moved images')
    parser.add_argument('--non-recursive', action='store_true',
                        help='Do not search subdirectories')
    
    args = parser.parse_args()
    
    process_dataset(
        args.directory,
        std_threshold=args.std_threshold,
        action=args.action,
        dest_dir=args.dest_dir,
        recursive=not args.non_recursive
    )

if __name__ == "__main__":
    main()