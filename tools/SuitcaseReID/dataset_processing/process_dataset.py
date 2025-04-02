#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import shutil
import random
import numpy as np
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process SuitcaseReID dataset')
    parser.add_argument('--dataset_path', type=str, default='/data1/zhaofanghan/OpenUnReID/datasets/SuitcaseReID_Multiview',
                        help='path to the original dataset')
    parser.add_argument('--output_path', type=str, default='/data1/zhaofanghan/OpenUnReID/datasets/SuitcaseReID_Processed',
                        help='path to save processed dataset')
    parser.add_argument('--min_images_per_viewpoint', type=int, default=5,
                        help='minimum number of images required per viewpoint')
    parser.add_argument('--single_viewpoint_augmentations', type=int, default=10,
                        help='number of augmentations for IDs with single viewpoint')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='ratio of data to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility')
    return parser.parse_args()

def setup_directories(output_path):
    """Create necessary directories for processed dataset in Market-1501 style structure"""
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'bounding_box_train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'query'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'bounding_box_test'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'stats'), exist_ok=True)

def analyze_dataset(dataset_path):
    """Analyze the dataset structure and extract ID and viewpoint information"""
    files = os.listdir(dataset_path)
    pattern = re.compile(r'(\d{4})_p_(\d+)_(\d+)\.jpg')
    
    id_to_files = defaultdict(list)
    id_to_viewpoints = defaultdict(set)
    viewpoint_counts = defaultdict(lambda: defaultdict(int))
    
    for file in files:
        match = pattern.match(file)
        if match:
            id_str, viewpoint, img_idx = match.groups()
            id_to_files[id_str].append(file)
            id_to_viewpoints[id_str].add(viewpoint)
            viewpoint_counts[id_str][viewpoint] += 1
    
    return id_to_files, id_to_viewpoints, viewpoint_counts

def reassign_ids(id_to_files):
    """Reassign IDs to ensure continuity while preserving original file structure"""
    old_to_new_id = {}
    for new_id, old_id in enumerate(sorted(id_to_files.keys()), 1):
        # Format to 4 digits with leading zeros
        old_to_new_id[old_id] = f"{new_id:04d}"
    
    return old_to_new_id

def create_airport_augmentations():
    """Create augmentations that simulate airport-like environments"""
    # Create a composition of augmentations simulating airport conditions
    return A.Compose([
        # Lighting variations (airports have varied lighting)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.HueSaturationValue(p=0.5),
        ], p=0.8),
        
        # Perspective and geometric variations (viewing angles in airport cameras)
        A.OneOf([
            A.Perspective(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), 
                    rotate=(-10, 10), shear=(-5, 5), p=0.5),
        ], p=0.7),
        
        # Noise and blur (security camera quality)
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.4),
        ], p=0.6),
        
        # Occlusion simulation (partial views in crowded airports)
        A.RandomCrop(height=384, width=256, p=0.4),
        
        # Compression artifacts (common in surveillance systems)
        A.ImageCompression(p=0.4),
    ])

def augment_image(image_path, transform, output_dir, new_filename):
    """Apply augmentation to an image and save it"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return False
        
    # Convert to RGB for augmentation
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply the augmentation
    augmented = transform(image=image)
    augmented_image = augmented['image']
    
    # Convert back to BGR for OpenCV
    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    
    # Save the augmented image
    output_path = os.path.join(output_dir, new_filename)
    cv2.imwrite(output_path, augmented_image)
    
    return True

def process_dataset(args):
    """Main function to process the dataset"""
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup directories
    setup_directories(args.output_path)
    
    # Analyze the dataset
    print("Analyzing dataset structure...")
    id_to_files, id_to_viewpoints, viewpoint_counts = analyze_dataset(args.dataset_path)
    
    # Reassign IDs
    print("Reassigning IDs...")
    old_to_new_id = reassign_ids(id_to_files)
    
    # Create augmentation transformations
    airport_transform = create_airport_augmentations()
    
    # Statistics tracking
    total_ids = len(id_to_files)
    original_image_count = sum(len(files) for files in id_to_files.values())
    augmented_image_count = 0
    images_per_id = defaultdict(int)
    viewpoints_per_id = defaultdict(int)
    
    # Split IDs into train and test
    all_ids = sorted(old_to_new_id.values())
    random.shuffle(all_ids)  # Shuffle to ensure random distribution
    num_train_ids = int(len(all_ids) * args.train_ratio)
    
    train_ids = set(all_ids[:num_train_ids])
    test_ids = set(all_ids[num_train_ids:])
    
    # Prepare to store processed images by ID
    processed_images_by_id = defaultdict(list)
    
    # Process each ID
    print("Processing dataset and performing augmentations...")
    for old_id, files in tqdm(id_to_files.items()):
        new_id = old_to_new_id[old_id]
        viewpoints = id_to_viewpoints[old_id]
        viewpoints_per_id[new_id] = len(viewpoints)
        
        # Determine destination directory based on ID
        dest_dir = 'bounding_box_train' if new_id in train_ids else 'bounding_box_test'
        
        # If ID has only one viewpoint, perform special augmentation
        if len(viewpoints) == 1:
            viewpoint = list(viewpoints)[0]
            viewpoint_files = [f for f in files if f"_p_{viewpoint}_" in f]
            
            # Copy original files
            for file in viewpoint_files:
                src_path = os.path.join(args.dataset_path, file)
                new_filename = file.replace(old_id, new_id)
                dst_path = os.path.join(args.output_path, dest_dir, new_filename)
                
                # Copy the file
                shutil.copy2(src_path, dst_path)
                processed_images_by_id[new_id].append(new_filename)
                images_per_id[new_id] += 1
                
                # For training IDs, perform augmentations
                if new_id in train_ids:
                    # Perform 10 augmentations for single-viewpoint IDs
                    for aug_idx in range(args.single_viewpoint_augmentations):
                        aug_filename = new_filename.replace('.jpg', f'_aug{aug_idx}.jpg')
                        success = augment_image(src_path, airport_transform, 
                                              os.path.join(args.output_path, dest_dir), aug_filename)
                        if success:
                            processed_images_by_id[new_id].append(aug_filename)
                            images_per_id[new_id] += 1
                            augmented_image_count += 1
        else:
            # Process multiple viewpoints
            for viewpoint in viewpoints:
                viewpoint_files = [f for f in files if f"_p_{viewpoint}_" in f]
                current_count = len(viewpoint_files)
                
                # Copy original files
                for file in viewpoint_files:
                    src_path = os.path.join(args.dataset_path, file)
                    new_filename = file.replace(old_id, new_id)
                    dst_path = os.path.join(args.output_path, dest_dir, new_filename)
                    
                    # Copy the file
                    shutil.copy2(src_path, dst_path)
                    processed_images_by_id[new_id].append(new_filename)
                    images_per_id[new_id] += 1
                
                # For training IDs, perform augmentations if necessary
                if new_id in train_ids and current_count < args.min_images_per_viewpoint:
                    needed_augmentations = args.min_images_per_viewpoint - current_count
                    
                    # Select random files to augment
                    for i in range(needed_augmentations):
                        # Choose a random file from this viewpoint
                        source_file = random.choice(viewpoint_files)
                        src_path = os.path.join(args.dataset_path, source_file)
                        
                        # Create augmented filename
                        aug_filename = source_file.replace(old_id, new_id).replace('.jpg', f'_aug{i}.jpg')
                        
                        # Apply augmentation
                        success = augment_image(src_path, airport_transform, 
                                               os.path.join(args.output_path, dest_dir), aug_filename)
                        if success:
                            processed_images_by_id[new_id].append(aug_filename)
                            images_per_id[new_id] += 1
                            augmented_image_count += 1
    
    # Create query set (one image per test ID)
    print("Creating query set...")
    query_count = 0
    for test_id in test_ids:
        if test_id in processed_images_by_id and processed_images_by_id[test_id]:
            # Select one image (preferably an original, not augmented image) for the query set
            query_candidates = [img for img in processed_images_by_id[test_id] if 'aug' not in img]
            
            # If no original images, use any available image
            if not query_candidates:
                query_candidates = processed_images_by_id[test_id]
            
            # Select a random image as the query
            query_image = random.choice(query_candidates)
            
            # Move from bounding_box_test to query
            src_path = os.path.join(args.output_path, 'bounding_box_test', query_image)
            dst_path = os.path.join(args.output_path, 'query', query_image)
            
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
                query_count += 1
    
    # Generate statistics
    print("Generating statistics...")
    stats = {
        "total_ids": total_ids,
        "total_train_ids": len(train_ids),
        "total_test_ids": len(test_ids),
        "original_images": original_image_count,
        "augmented_images": augmented_image_count,
        "total_images": original_image_count + augmented_image_count,
        "query_images": query_count
    }
    
    # Count actual images in each directory
    train_count = len(os.listdir(os.path.join(args.output_path, 'bounding_box_train')))
    test_count = len(os.listdir(os.path.join(args.output_path, 'bounding_box_test')))
    query_count = len(os.listdir(os.path.join(args.output_path, 'query')))
    
    # Save statistics
    with open(os.path.join(args.output_path, 'stats', 'dataset_stats.txt'), 'w') as f:
        f.write("SuitcaseReID Dataset Statistics\n")
        f.write("============================\n\n")
        f.write(f"Total IDs: {stats['total_ids']}\n")
        f.write(f"Train IDs: {stats['total_train_ids']}\n")
        f.write(f"Test IDs: {stats['total_test_ids']}\n")
        f.write(f"Original Images: {stats['original_images']}\n")
        f.write(f"Augmented Images: {stats['augmented_images']}\n")
        f.write(f"Total Images: {stats['total_images']}\n")
        f.write(f"Training Images (bounding_box_train): {train_count}\n")
        f.write(f"Test Gallery Images (bounding_box_test): {test_count}\n")
        f.write(f"Query Images: {query_count}\n\n")
        
        f.write("ID Distribution:\n")
        for id_str, count in sorted(images_per_id.items()):
            viewpoint_count = viewpoints_per_id[id_str]
            f.write(f"ID {id_str}: {count} images across {viewpoint_count} viewpoints\n")
    
    # Generate visualizations
    plt.figure(figsize=(10, 6))
    plt.hist([count for count in images_per_id.values()], bins=20)
    plt.xlabel("Images per ID")
    plt.ylabel("Frequency")
    plt.title("Distribution of Images per ID")
    plt.savefig(os.path.join(args.output_path, 'stats', 'images_per_id.png'))
    
    plt.figure(figsize=(10, 6))
    plt.hist([count for count in viewpoints_per_id.values()], bins=4)
    plt.xlabel("Viewpoints per ID")
    plt.ylabel("Frequency")
    plt.title("Distribution of Viewpoints per ID")
    plt.xticks([1, 2, 3, 4])
    plt.savefig(os.path.join(args.output_path, 'stats', 'viewpoints_per_id.png'))
    
    # Create a detailed CSV report
    df = pd.DataFrame({
        'ID': list(images_per_id.keys()),
        'Images': [images_per_id[id_str] for id_str in images_per_id.keys()],
        'Viewpoints': [viewpoints_per_id[id_str] for id_str in images_per_id.keys()],
        'Split': ['Train' if id_str in train_ids else 'Test' for id_str in images_per_id.keys()]
    })
    df.to_csv(os.path.join(args.output_path, 'stats', 'id_details.csv'), index=False)
    
    print(f"Processing completed. Results saved to {args.output_path}")
    print(f"Statistics available in {os.path.join(args.output_path, 'stats')}")
    print(f"Market-1501 style directory structure created:")
    print(f"  - bounding_box_train: {train_count} images")
    print(f"  - bounding_box_test: {test_count} images")
    print(f"  - query: {query_count} images (one per test ID)")
    
    return stats

if __name__ == "__main__":
    args = parse_args()
    process_dataset(args)