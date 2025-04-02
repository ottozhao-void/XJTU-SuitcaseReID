#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from collections import defaultdict
import seaborn as sns
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize processed SuitcaseReID dataset')
    parser.add_argument('--processed_path', type=str, default='/data1/zhaofanghan/OpenUnReID/datasets/SuitcaseReID_Processed',
                        help='path to the processed dataset')
    parser.add_argument('--visualization_samples', type=int, default=5,
                        help='number of IDs to sample for visualizations')
    parser.add_argument('--output_path', type=str, default=None,
                        help='path to save visualizations, defaults to processed_path/visualization')
    return parser.parse_args()

def setup_output_directory(args):
    """Setup output directory for visualizations"""
    if args.output_path is None:
        args.output_path = os.path.join(args.processed_path, 'visualization')
    
    os.makedirs(args.output_path, exist_ok=True)
    return args.output_path

def analyze_dataset_splits(processed_path):
    """Analyze train/test/query splits of the dataset in Market-1501 format"""
    train_files = os.listdir(os.path.join(processed_path, 'bounding_box_train'))
    test_files = os.listdir(os.path.join(processed_path, 'bounding_box_test'))
    query_files = os.listdir(os.path.join(processed_path, 'query'))
    
    pattern = re.compile(r'(\d{4})_p_(\d+)_(.*?)\.jpg')
    
    # Analyze train set
    train_ids = set()
    train_viewpoints = defaultdict(set)
    train_images_per_id = defaultdict(int)
    train_augmented = 0
    
    for file in train_files:
        match = pattern.match(file)
        if match:
            id_str, viewpoint, img_idx = match.groups()
            train_ids.add(id_str)
            train_viewpoints[id_str].add(viewpoint)
            train_images_per_id[id_str] += 1
            if 'aug' in img_idx:
                train_augmented += 1
    
    # Analyze test gallery set
    test_ids = set()
    test_viewpoints = defaultdict(set)
    test_images_per_id = defaultdict(int)
    test_augmented = 0
    
    for file in test_files:
        match = pattern.match(file)
        if match:
            id_str, viewpoint, img_idx = match.groups()
            test_ids.add(id_str)
            test_viewpoints[id_str].add(viewpoint)
            test_images_per_id[id_str] += 1
            if 'aug' in img_idx:
                test_augmented += 1
    
    # Analyze query set
    query_ids = set()
    query_viewpoints = defaultdict(set)
    query_images_per_id = defaultdict(int)
    
    for file in query_files:
        match = pattern.match(file)
        if match:
            id_str, viewpoint, img_idx = match.groups()
            query_ids.add(id_str)
            query_viewpoints[id_str].add(viewpoint)
            query_images_per_id[id_str] += 1
    
    # Combine test and query IDs (they should be the same set)
    all_test_ids = test_ids.union(query_ids)
    
    # Compute statistics
    stats = {
        'train_ids': len(train_ids),
        'test_ids': len(all_test_ids),
        'train_images': len(train_files),
        'test_gallery_images': len(test_files),
        'query_images': len(query_files),
        'train_augmented': train_augmented,
        'test_augmented': test_augmented,
        'train_images_per_id': dict(train_images_per_id),
        'test_gallery_images_per_id': dict(test_images_per_id),
        'query_images_per_id': dict(query_images_per_id),
        'train_viewpoints': {id_str: len(viewpoints) for id_str, viewpoints in train_viewpoints.items()},
        'test_viewpoints': {id_str: len(viewpoints) for id_str, viewpoints in test_viewpoints.items()},
        'query_viewpoints': {id_str: len(viewpoints) for id_str, viewpoints in query_viewpoints.items()}
    }
    
    return stats, train_ids, all_test_ids

def create_split_visualizations(stats, output_path):
    """Create visualizations for train/test splits"""
    # Train/Test ID distribution
    plt.figure(figsize=(15, 5))
    
    # Images per ID distribution
    plt.subplot(1, 3, 1)
    train_images = list(stats['train_images_per_id'].values())
    test_images = [stats['test_gallery_images_per_id'].get(id_str, 0) + 
                  stats['query_images_per_id'].get(id_str, 0) 
                  for id_str in set(stats['test_gallery_images_per_id']).union(stats['query_images_per_id'])]
    
    plt.hist([train_images, test_images], bins=15, alpha=0.7, 
             label=['Train', 'Test'], color=['blue', 'orange'])
    plt.xlabel("Images per ID")
    plt.ylabel("Frequency")
    plt.title("Distribution of Images per ID")
    plt.legend()
    
    # Viewpoints per ID distribution
    plt.subplot(1, 3, 2)
    train_viewpoints = list(stats['train_viewpoints'].values())
    test_viewpoints = list(stats['test_viewpoints'].values())
    
    plt.hist([train_viewpoints, test_viewpoints], bins=4, alpha=0.7,
             label=['Train', 'Test'], color=['blue', 'orange'])
    plt.xlabel("Viewpoints per ID")
    plt.ylabel("Frequency")
    plt.title("Distribution of Viewpoints per ID")
    plt.xticks([1, 2, 3, 4])
    plt.legend()
    
    # Query distribution
    plt.subplot(1, 3, 3)
    query_counts = list(stats['query_images_per_id'].values())
    
    plt.hist(query_counts, bins=range(1, 4), alpha=0.7, color='green')
    plt.xlabel("Query Images per ID")
    plt.ylabel("Frequency")
    plt.title("Query Images Distribution")
    plt.xticks([1, 2, 3])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'dataset_distribution.png'))

def visualize_augmentations(processed_path, output_path, sample_ids):
    """Visualize original images vs augmented images for sample IDs"""
    for id_str in sample_ids:
        # Check if this ID is in the training set
        train_dir = os.path.join(processed_path, 'bounding_box_train')
        id_files = [f for f in os.listdir(train_dir) if f.startswith(id_str)]
        
        if not id_files:  # ID not in training set
            continue
            
        # Group by viewpoint
        viewpoints = {}
        for file in id_files:
            match = re.match(r'\d{4}_p_(\d+)_(.*?)\.jpg', file)
            if match:
                viewpoint, img_idx = match.groups()
                if viewpoint not in viewpoints:
                    viewpoints[viewpoint] = {'original': [], 'augmented': []}
                
                if 'aug' in img_idx:
                    viewpoints[viewpoint]['augmented'].append(file)
                else:
                    viewpoints[viewpoint]['original'].append(file)
        
        # Create visualization for each viewpoint
        for viewpoint, images in viewpoints.items():
            if images['augmented']:  # Only visualize if there are augmentations
                # Determine how many images to display
                n_orig = min(3, len(images['original']))
                n_aug = min(5, len(images['augmented']))
                
                fig, axes = plt.subplots(2, max(n_orig, n_aug), figsize=(15, 6))
                
                # Plot original images
                for i in range(n_orig):
                    if i < len(images['original']):
                        img = cv2.imread(os.path.join(train_dir, images['original'][i]))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        axes[0, i].imshow(img)
                        axes[0, i].set_title(f"Original {i+1}")
                        axes[0, i].axis('off')
                
                # Plot augmented images
                for i in range(n_aug):
                    if i < len(images['augmented']):
                        img = cv2.imread(os.path.join(train_dir, images['augmented'][i]))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        axes[1, i].imshow(img)
                        axes[1, i].set_title(f"Augmented {i+1}")
                        axes[1, i].axis('off')
                
                # Turn off any unused subplots
                for i in range(n_orig, max(n_orig, n_aug)):
                    axes[0, i].axis('off')
                for i in range(n_aug, max(n_orig, n_aug)):
                    axes[1, i].axis('off')
                
                plt.suptitle(f"ID: {id_str}, Viewpoint: {viewpoint} (training set)")
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, f"{id_str}_vp{viewpoint}_augmentations.png"))
                plt.close()

def visualize_query_gallery(processed_path, output_path, sample_ids, max_samples=3):
    """Visualize query images alongside gallery images for the same ID"""
    query_dir = os.path.join(processed_path, 'query')
    gallery_dir = os.path.join(processed_path, 'bounding_box_test')
    
    # Filter to only include IDs that are in the test set
    test_ids = []
    for id_str in sample_ids:
        query_files = [f for f in os.listdir(query_dir) if f.startswith(id_str)]
        if query_files:
            test_ids.append(id_str)
    
    # Cap at max_samples
    test_ids = test_ids[:max_samples]
    
    for id_str in test_ids:
        query_files = [f for f in os.listdir(query_dir) if f.startswith(id_str)]
        gallery_files = [f for f in os.listdir(gallery_dir) if f.startswith(id_str)]
        
        if not query_files or not gallery_files:
            continue
            
        # Sort files to get consistent results
        query_files.sort()
        gallery_files.sort()
        
        # Limit the number of gallery images to display
        max_gallery = min(5, len(gallery_files))
        
        fig = plt.figure(figsize=(12, 4))
        
        # Plot query image(s)
        for i, file in enumerate(query_files[:1]):  # Just show the first query image if multiple exist
            ax = plt.subplot(1, max_gallery + 1, 1)
            img = cv2.imread(os.path.join(query_dir, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title("Query", fontsize=12)
            plt.axis('off')
        
        # Plot gallery images
        for i, file in enumerate(gallery_files[:max_gallery]):
            ax = plt.subplot(1, max_gallery + 1, i + 2)
            img = cv2.imread(os.path.join(gallery_dir, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(f"Gallery {i+1}", fontsize=12)
            plt.axis('off')
        
        plt.suptitle(f"ID: {id_str} - Query and Gallery Images")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{id_str}_query_gallery.png"))
        plt.close()

def verify_requirements(processed_path):
    """Verify that all viewpoints have at least 5 images as required"""
    train_files = os.listdir(os.path.join(processed_path, 'bounding_box_train'))
    test_files = os.listdir(os.path.join(processed_path, 'bounding_box_test'))
    
    # Collect viewpoint counts
    train_viewpoints = defaultdict(lambda: defaultdict(int))
    test_viewpoints = defaultdict(lambda: defaultdict(int))
    
    for file in train_files:
        match = re.match(r'(\d{4})_p_(\d+)_.*?\.jpg', file)
        if match:
            id_str, viewpoint = match.groups()
            train_viewpoints[id_str][viewpoint] += 1
    
    for file in test_files:
        match = re.match(r'(\d{4})_p_(\d+)_.*?\.jpg', file)
        if match:
            id_str, viewpoint = match.groups()
            test_viewpoints[id_str][viewpoint] += 1
    
    # Check requirements
    train_issues = []
    for id_str, viewpoints in train_viewpoints.items():
        for viewpoint, count in viewpoints.items():
            if count < 5:
                train_issues.append(f"ID {id_str}, Viewpoint {viewpoint}: Only {count} images (needs 5+)")
    
    test_issues = []
    for id_str, viewpoints in test_viewpoints.items():
        for viewpoint, count in viewpoints.items():
            if count < 5:
                test_issues.append(f"ID {id_str}, Viewpoint {viewpoint}: Only {count} images (needs 5+)")
    
    return train_issues, test_issues

def main():
    args = parse_args()
    output_path = setup_output_directory(args)
    
    print("Analyzing dataset splits...")
    stats, train_ids, test_ids = analyze_dataset_splits(args.processed_path)
    
    print("Creating visualizations for dataset splits...")
    create_split_visualizations(stats, output_path)
    
    print("Verifying dataset requirements...")
    train_issues, test_issues = verify_requirements(args.processed_path)
    
    if train_issues:
        print(f"\nWARNING: Found {len(train_issues)} viewpoints in train set with fewer than 5 images:")
        for issue in train_issues[:10]:  # Show only first 10 to avoid clutter
            print(f"  - {issue}")
        if len(train_issues) > 10:
            print(f"  ... and {len(train_issues) - 10} more issues.")
    else:
        print("\nAll train set viewpoints meet the minimum requirement of 5 images.")
    
    if test_issues:
        print(f"\nNOTE: Found {len(test_issues)} viewpoints in test gallery set with fewer than 5 images.")
        print("(This is expected as we don't augment the test set)")
    
    # Sample IDs for visualizations
    print("\nCreating visualizations of augmentations...")
    all_ids = list(train_ids.union(test_ids))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(all_ids)
    sample_ids = all_ids[:args.visualization_samples]
    
    visualize_augmentations(args.processed_path, output_path, sample_ids)
    visualize_query_gallery(args.processed_path, output_path, sample_ids)
    
    # Generate summary report
    print("\nGenerating summary report...")
    with open(os.path.join(output_path, 'verification_report.txt'), 'w') as f:
        f.write("SuitcaseReID Dataset Verification Report\n")
        f.write("=====================================\n\n")
        
        f.write("Dataset Summary (Market-1501 format)\n")
        f.write("--------------------------------\n")
        f.write(f"Total IDs: {stats['train_ids'] + stats['test_ids']}\n")
        f.write(f"Train IDs: {stats['train_ids']}\n")
        f.write(f"Test IDs: {stats['test_ids']}\n")
        f.write(f"Training Images (bounding_box_train): {stats['train_images']} (including {stats['train_augmented']} augmented)\n")
        f.write(f"Test Gallery Images (bounding_box_test): {stats['test_gallery_images']}\n")
        f.write(f"Query Images (query): {stats['query_images']}\n\n")
        
        f.write("Requirements Verification\n")
        f.write("------------------------\n")
        if train_issues:
            f.write(f"Train set issues: {len(train_issues)} viewpoints have fewer than 5 images\n")
            for issue in train_issues[:20]:
                f.write(f"  - {issue}\n")
            if len(train_issues) > 20:
                f.write(f"  ... and {len(train_issues) - 20} more issues.\n")
        else:
            f.write("Train set: All viewpoints meet requirement of 5+ images\n")
        
        f.write("\nAverage images per ID:\n")
        if stats['train_images_per_id']:
            f.write(f"  Train: {np.mean(list(stats['train_images_per_id'].values())):.2f}\n")
        if stats['test_gallery_images_per_id']:
            f.write(f"  Test Gallery: {np.mean(list(stats['test_gallery_images_per_id'].values())):.2f}\n")
        if stats['query_images_per_id']:
            f.write(f"  Query: {np.mean(list(stats['query_images_per_id'].values())):.2f}\n")
    
    print(f"\nVerification complete. Report saved to {os.path.join(output_path, 'verification_report.txt')}")
    print(f"Visualization images saved to {output_path}")

if __name__ == "__main__":
    main()