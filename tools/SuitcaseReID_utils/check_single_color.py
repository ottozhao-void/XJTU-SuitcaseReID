#!/usr/bin/env python
"""
Script to check for and remove images that are predominantly single-colored (>80% of pixels)
Using GPU acceleration for faster processing
"""

import os
import re
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn.functional as F

def is_predominantly_single_color(img_path, threshold_percent=80, max_color_distance=10, device='cuda:3'):
    """
    Check if an image is predominantly a single color (>80% pixels very similar)
    Uses GPU acceleration for faster processing
    
    Args:
        img_path: Path to the image
        threshold_percent: Percentage threshold to consider as "predominantly" single color
        max_color_distance: Maximum distance between colors to consider them similar
        device: GPU device to use for processing
        
    Returns:
        (is_predominant, percentage, dominant_color): Tuple with boolean, percentage of dominant color, and the dominant RGB
    """
    try:
        # Open image and convert to tensor
        img = Image.open(img_path)
        data = np.array(img)
        
        # If image is grayscale, convert to RGB
        if len(data.shape) == 2:
            return (True, 100.0, (data[0, 0], data[0, 0], data[0, 0]))
        
        # Convert to PyTorch tensor and move to GPU
        data_tensor = torch.from_numpy(data).to(device)
        
        # Reshape the tensor to a list of pixels
        pixels = data_tensor.reshape(-1, data_tensor.shape[2])
        total_pixels = pixels.shape[0]
        
        # Bin the colors
        binned_pixels = (pixels // max_color_distance).to(torch.int64)
        
        # Use torch.unique with return_counts to efficiently find color frequencies
        unique_colors, counts = torch.unique(binned_pixels, dim=0, return_counts=True)
        
        # Find the most common color group
        max_idx = torch.argmax(counts)
        most_common_count = counts[max_idx].item()
        most_common_key = unique_colors[max_idx].cpu().numpy()
        
        percentage = (most_common_count / total_pixels) * 100
        
        # Get an example of this dominant color
        dominant_color = (
            most_common_key[0] * max_color_distance + max_color_distance // 2,
            most_common_key[1] * max_color_distance + max_color_distance // 2,
            most_common_key[2] * max_color_distance + max_color_distance // 2
        )
        
        # Check if percentage exceeds threshold
        return (percentage >= threshold_percent, percentage, dominant_color)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return (False, 0, (0, 0, 0))
    
def extract_prefix(filename):
    """Extract the prefix pattern (e.g., '0010_p_3') from a filename"""
    match = re.match(r'^(\d{4}_p_\d+).*\.jpg$', filename)
    return match.group(1) if match else None

def check_and_remove_single_color(directory, threshold_percent=80, max_color_distance=10, device='cuda:3', dry_run=False):
    """
    Check for and remove images that are predominantly single-colored
    
    Args:
        directory: Directory containing the images
        threshold_percent: Percentage threshold to consider as "predominantly" single color
        max_color_distance: Maximum distance between colors to consider them similar
        device: GPU device to use for processing
        dry_run: If True, only report findings without removing files
    """
    print(f"Scanning for predominantly single-color images in {directory} using GPU ({device})...")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Falling back to CPU processing, which will be slower.")
        device = 'cpu'
    
    total_images = 0
    problematic_images = []
    affected_prefixes = set()
    
    for filename in tqdm(os.listdir(directory)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            total_images += 1
            img_path = os.path.join(directory, filename)
            is_predominant, percentage, dominant_color = is_predominantly_single_color(
                img_path, threshold_percent, max_color_distance, device
            )
            
            if is_predominant:
                problematic_images.append((img_path, percentage, dominant_color))
                
                # Extract and record the prefix
                prefix = extract_prefix(filename)
                if prefix:
                    affected_prefixes.add(prefix)
    
    # Print summary
    print(f"\nFound {len(problematic_images)} images that are predominantly single-colored out of {total_images} total images.")
    print(f"Threshold: {threshold_percent}% of pixels with similar color (max distance: {max_color_distance})")
    
    if problematic_images:
        print("\nPredominantly single-colored images found:")
        for img_path, percentage, dominant_color in problematic_images:
            print(f"{os.path.basename(img_path)}: {percentage:.1f}% pixels similar to RGB{dominant_color}")
        
        print("\nAffected prefixes (for reindexing):")
        print(", ".join(sorted(affected_prefixes)))
        
        # Save the affected prefixes to a file
        prefixes_file = os.path.join(os.path.dirname(directory), "affected_prefixes.txt")
        with open(prefixes_file, "w") as f:
            for prefix in sorted(affected_prefixes):
                f.write(f"{prefix}\n")
        print(f"Saved affected prefixes to {prefixes_file}")
        
        if not dry_run:
            print("\nRemoving predominantly single-colored images...")
            for img_path, _, _ in problematic_images:
                os.remove(img_path)
            print(f"Removed {len(problematic_images)} problematic images.")
        else:
            print("\nThis was a dry run. No files were removed.")
    else:
        print("No predominantly single-colored images found.")

def main():
    parser = argparse.ArgumentParser(description="Check for and remove images that are predominantly single-colored")
    parser.add_argument('directory', help='Directory containing the image dataset')
    parser.add_argument('--threshold', type=float, default=80.0,
                        help='Percentage threshold to consider as "predominantly" single color (default: 80.0)')
    parser.add_argument('--color-distance', type=int, default=10,
                        help='Maximum color distance to consider colors similar (default: 10)')
    parser.add_argument('--device', type=str, default='cuda:3',
                        help='GPU device to use (default: cuda:3)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without actually removing files')
    
    args = parser.parse_args()
    
    check_and_remove_single_color(
        args.directory, 
        threshold_percent=args.threshold,
        max_color_distance=args.color_distance,
        device=args.device,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()