#!/usr/bin/env python
"""
MVB Dataset Analysis Script

This script analyzes the MVB dataset structure and properties and creates
visualizations to better understand its characteristics.

Created: April 10, 2025
"""

import os
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.gridspec as gridspec
from PIL import Image
import random

# Set seaborn style for better visualizations
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Paths
train_info_path = "/data1/zhaofanghan/SuitcaseReID/OpenUnReID/datasets/MVB/MVB_train/Info"
val_info_path = "/data1/zhaofanghan/SuitcaseReID/OpenUnReID/datasets/MVB/MVB_val/Info"
train_format_json = os.path.join(train_info_path, "format.json")
val_gallery_json = os.path.join(val_info_path, "val_gallery.json")
val_probe_json = os.path.join(val_info_path, "val_probe.json")
output_dir = "/data1/zhaofanghan/SuitcaseReID/OpenUnReID/tools/MVB_supervised/analysis_results"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def load_json_data(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_id_from_filename(filename):
    """Extract the suitcase ID from a filename"""
    match = re.match(r'^(\d+)_', filename)
    if match:
        return match.group(1)
    return None

def extract_datatype_from_filename(filename):
    """Extract whether the image is from gallery (g) or probe (p)"""
    match = re.search(r'_([gp])_', filename)
    if match:
        return match.group(1)
    return None

def analyze_train_data():
    """Analyze the training dataset"""
    print("Analyzing training dataset...")
    train_data = load_json_data(train_format_json)
    
    # Initialize data structures
    materials = defaultdict(int)
    datasource_count = {'g': 0, 'p': 0}
    view_angles = {'g': defaultdict(int), 'p': defaultdict(int)}
    id_to_images = defaultdict(list)
    image_dims = []
    
    # Process each image
    for img in train_data['image']:
        # Count materials
        material = img.get('material', 'unknown')
        materials[material] += 1
        
        # Count data sources
        datatype = img.get('datatype', 'unknown')
        datasource_count[datatype] += 1
        
        # Extract suitcase ID
        suitcase_id = img.get('id', 'unknown')
        id_to_images[suitcase_id].append(img)
        
        # Extract view angle
        img_name = img.get('image_name', '')
        view_match = re.search(r'_\d+\.jpg$', img_name)
        if view_match:
            view = view_match.group(0)[1:-4]  # Extract the view number
            view_angles[datatype][view] += 1
        
        # Record image dimensions
        dims = img.get('dimsize', [0, 0])
        image_dims.append(dims)
    
    # Calculate statistics
    num_ids = len(id_to_images)
    total_images = len(train_data['image'])
    avg_images_per_id = total_images / num_ids if num_ids > 0 else 0
    
    # Count images per ID
    images_per_id = [len(images) for _, images in id_to_images.items()]
    
    print(f"Number of unique suitcase IDs: {num_ids}")
    print(f"Total number of images: {total_images}")
    print(f"Average images per ID: {avg_images_per_id:.2f}")
    
    # Create material distribution plot
    plt.figure(figsize=(10, 6))
    materials_df = pd.Series(materials).reset_index()
    materials_df.columns = ['Material', 'Count']
    plt.title("Distribution of Suitcase Materials (Training Set)", fontsize=15)
    ax = sns.barplot(data=materials_df, x='Material', y='Count', palette='viridis')
    for i, count in enumerate(materials_df['Count']):
        ax.text(i, count + 50, f"{count} ({count/total_images*100:.1f}%)", 
                ha='center', va='bottom', fontsize=10)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xlabel("Material Type", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_materials_distribution.png"), dpi=300)
    plt.close()
    
    # Create data source distribution plot
    plt.figure(figsize=(8, 6))
    labels = ['Pre-check (gallery)', 'Post-check (probe)']
    counts = [datasource_count['g'], datasource_count['p']]
    plt.title("Distribution of Image Sources (Training Set)", fontsize=15)
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_datasource_distribution.png"), dpi=300)
    plt.close()
    
    # Create images per ID distribution
    plt.figure(figsize=(10, 6))
    plt.title("Distribution of Images per Suitcase ID (Training Set)", fontsize=15)
    sns.histplot(images_per_id, bins=20, kde=True)
    plt.xlabel("Number of Images per Suitcase ID", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_images_per_id_distribution.png"), dpi=300)
    plt.close()
    
    # Create view angle distribution
    plt.figure(figsize=(12, 6))
    plt.title("Distribution of View Angles by Source Type (Training Set)", fontsize=15)
    
    # Convert to DataFrame for easier plotting
    g_views = pd.Series(view_angles['g']).reset_index()
    g_views.columns = ['View', 'Count']
    g_views['Source'] = 'Pre-check (gallery)'
    
    p_views = pd.Series(view_angles['p']).reset_index()
    p_views.columns = ['View', 'Count']
    p_views['Source'] = 'Post-check (probe)'
    
    views_df = pd.concat([g_views, p_views])
    
    sns.barplot(data=views_df, x='View', y='Count', hue='Source', palette=['#ff9999','#66b3ff'])
    plt.xlabel("View Angle", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(title="Source Type")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_view_angle_distribution.png"), dpi=300)
    plt.close()
    
    # Create image dimensions scatter plot
    plt.figure(figsize=(10, 6))
    plt.title("Image Dimensions Distribution (Training Set)", fontsize=15)
    widths = [dim[0] for dim in image_dims]
    heights = [dim[1] for dim in image_dims]
    plt.scatter(widths, heights, alpha=0.3, color='blue')
    plt.xlabel("Width (pixels)", fontsize=12)
    plt.ylabel("Height (pixels)", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_image_dimensions.png"), dpi=300)
    plt.close()
    
    # Material distribution by source type
    material_by_source = defaultdict(lambda: {'g': 0, 'p': 0})
    for img in train_data['image']:
        material = img.get('material', 'unknown')
        datatype = img.get('datatype', 'unknown')
        material_by_source[material][datatype] += 1
    
    # Create material by source plot
    plt.figure(figsize=(12, 7))
    materials_list = list(material_by_source.keys())
    g_counts = [material_by_source[m]['g'] for m in materials_list]
    p_counts = [material_by_source[m]['p'] for m in materials_list]
    
    index = np.arange(len(materials_list))
    bar_width = 0.35
    
    plt.bar(index, g_counts, bar_width, label='Pre-check (gallery)', color='#ff9999')
    plt.bar(index + bar_width, p_counts, bar_width, label='Post-check (probe)', color='#66b3ff')
    
    plt.xlabel('Material Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Material Distribution by Source Type (Training Set)', fontsize=15)
    plt.xticks(index + bar_width/2, materials_list)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_material_by_source.png"), dpi=300)
    plt.close()
    
    return {
        'num_ids': num_ids,
        'total_images': total_images,
        'avg_images_per_id': avg_images_per_id,
        'materials': materials,
        'datasource_count': datasource_count,
        'view_angles': view_angles,
        'images_per_id': images_per_id
    }

def analyze_val_data():
    """Analyze the validation dataset"""
    print("Analyzing validation dataset...")
    val_gallery_data = load_json_data(val_gallery_json)
    val_probe_data = load_json_data(val_probe_json)
    
    # Count data
    gallery_count = len(val_gallery_data['image'])
    probe_count = len(val_probe_data['image'])
    
    # Extract IDs from gallery
    gallery_ids = set()
    for img in val_gallery_data['image']:
        id_str = img.get('id')
        if id_str:
            gallery_ids.add(id_str)
    
    num_ids = len(gallery_ids)
    total_images = gallery_count + probe_count
    avg_gallery_per_id = gallery_count / num_ids if num_ids > 0 else 0
    avg_probe_per_id = probe_count / num_ids if num_ids > 0 else 0
    
    print(f"Number of unique suitcase IDs in validation: {num_ids}")
    print(f"Total number of gallery images: {gallery_count}")
    print(f"Total number of probe images: {probe_count}")
    print(f"Average gallery images per ID: {avg_gallery_per_id:.2f}")
    print(f"Average probe images per ID: {avg_probe_per_id:.2f}")
    
    # Create a pie chart showing gallery vs probe distribution
    plt.figure(figsize=(8, 6))
    labels = ['Gallery (Pre-check)', 'Probe (Post-check)']
    counts = [gallery_count, probe_count]
    plt.title("Distribution of Gallery vs Probe Images (Validation Set)", fontsize=15)
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "val_gallery_probe_distribution.png"), dpi=300)
    plt.close()
    
    # Count view angles in gallery
    gallery_view_angles = defaultdict(int)
    for img in val_gallery_data['image']:
        img_name = img.get('image_name', '')
        view_match = re.search(r'_(\d+)\.jpg$', img_name)
        if view_match:
            view = view_match.group(1)
            gallery_view_angles[view] += 1
    
    # Create view angle distribution for gallery
    plt.figure(figsize=(10, 6))
    plt.title("Distribution of View Angles in Gallery (Validation Set)", fontsize=15)
    view_df = pd.Series(gallery_view_angles).reset_index()
    view_df.columns = ['View Angle', 'Count']
    
    sns.barplot(data=view_df, x='View Angle', y='Count', color='#ff9999')
    plt.xlabel("View Angle", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "val_gallery_view_angle_distribution.png"), dpi=300)
    plt.close()
    
    return {
        'num_ids': num_ids,
        'gallery_count': gallery_count,
        'probe_count': probe_count,
        'total_images': total_images,
        'avg_gallery_per_id': avg_gallery_per_id,
        'avg_probe_per_id': avg_probe_per_id,
        'gallery_view_angles': gallery_view_angles
    }

def compare_train_val():
    """Compare training and validation datasets"""
    train_stats = analyze_train_data()
    val_stats = analyze_val_data()
    
    # Create a comparison table
    comparison_data = {
        'Metric': [
            'Number of Unique IDs',
            'Total Images',
            'Gallery/Pre-check Images',
            'Probe/Post-check Images',
            'Avg. Images per ID'
        ],
        'Training Set': [
            train_stats['num_ids'],
            train_stats['total_images'],
            train_stats['datasource_count']['g'],
            train_stats['datasource_count']['p'],
            f"{train_stats['avg_images_per_id']:.2f}"
        ],
        'Validation Set': [
            val_stats['num_ids'],
            val_stats['total_images'],
            val_stats['gallery_count'],
            val_stats['probe_count'],
            f"{(val_stats['avg_gallery_per_id'] + val_stats['avg_probe_per_id']):.2f}"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Create a summary table figure
    plt.figure(figsize=(12, 6))
    plt.title("Comparison of Training and Validation Sets", fontsize=15)
    
    # Turn off axis
    plt.axis('off')
    
    # Create a table
    the_table = plt.table(
        cellText=df.values, 
        colLabels=df.columns, 
        loc='center', 
        cellLoc='center', 
        colWidths=[0.5, 0.25, 0.25]
    )
    
    # Style the table
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1.5, 2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_val_comparison.png"), dpi=300)
    plt.close()
    
    # Create distribution comparison
    categories = ['Gallery/Pre-check', 'Probe/Post-check']
    train_values = [train_stats['datasource_count']['g'], train_stats['datasource_count']['p']]
    val_values = [val_stats['gallery_count'], val_stats['probe_count']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, train_values, width, label='Training Set')
    rects2 = ax.bar(x + width/2, val_values, width, label='Validation Set')
    
    ax.set_title('Comparison of Image Types between Training and Validation Sets', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add value labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_val_distribution_comparison.png"), dpi=300)
    plt.close()
    
    # Create a combined summary plot with multiple visualizations
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
    
    # Dataset size comparison
    ax1 = plt.subplot(gs[0, 0])
    labels = ['Training Set', 'Validation Set']
    sizes = [train_stats['total_images'], val_stats['total_images']]
    colors = ['#ff9999', '#66b3ff']
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.axis('equal')
    ax1.set_title('Dataset Size Comparison', fontsize=14)
    
    # Number of unique IDs comparison
    ax2 = plt.subplot(gs[0, 1])
    id_counts = [train_stats['num_ids'], val_stats['num_ids']]
    ax2.bar(labels, id_counts, color=colors)
    ax2.set_title('Unique Suitcase IDs', fontsize=14)
    for i, v in enumerate(id_counts):
        ax2.text(i, v + 50, str(v), ha='center')
    
    # Image type distribution in Training
    ax3 = plt.subplot(gs[1, 0])
    train_dist = [
        train_stats['datasource_count']['g'], 
        train_stats['datasource_count']['p']
    ]
    train_labels = ['Pre-check\n(Gallery)', 'Post-check\n(Probe)']
    ax3.pie(train_dist, labels=train_labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    ax3.axis('equal')
    ax3.set_title('Training Set Distribution', fontsize=14)
    
    # Image type distribution in Validation
    ax4 = plt.subplot(gs[1, 1])
    val_dist = [val_stats['gallery_count'], val_stats['probe_count']]
    val_labels = ['Pre-check\n(Gallery)', 'Post-check\n(Probe)']
    ax4.pie(val_dist, labels=val_labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    ax4.axis('equal')
    ax4.set_title('Validation Set Distribution', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mvb_dataset_summary.png"), dpi=300)
    plt.close()
    
    # Create a comprehensive report
    report_path = os.path.join(output_dir, "mvb_dataset_analysis.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# MVB Dataset Analysis Report\n\n")
        f.write("## Dataset Overview\n\n")
        f.write("The MVB (Multi-View Baggage) dataset consists of images of suitcases/luggage taken at an airport, ")
        f.write("with images captured at both pre-check and post-check locations from multiple viewing angles.\n\n")
        
        f.write("### Summary Statistics\n\n")
        f.write("| Metric | Training Set | Validation Set |\n")
        f.write("|--------|-------------|---------------|\n")
        for i in range(len(comparison_data['Metric'])):
            f.write(f"| {comparison_data['Metric'][i]} | {comparison_data['Training Set'][i]} | {comparison_data['Validation Set'][i]} |\n")
        
        f.write("\n## Training Set Analysis\n\n")
        f.write(f"- The training set contains {train_stats['num_ids']} unique suitcase IDs with a total of {train_stats['total_images']} images.\n")
        f.write(f"- On average, each suitcase ID has {train_stats['avg_images_per_id']:.2f} images.\n")
        f.write(f"- There are {train_stats['datasource_count']['g']} images from pre-check (gallery) and {train_stats['datasource_count']['p']} from post-check (probe).\n\n")
        
        f.write("### Material Distribution\n\n")
        f.write("| Material | Count | Percentage |\n")
        f.write("|----------|-------|------------|\n")
        total = sum(train_stats['materials'].values())
        for material, count in sorted(train_stats['materials'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            f.write(f"| {material} | {count} | {percentage:.2f}% |\n")
        
        f.write("\n## Validation Set Analysis\n\n")
        f.write(f"- The validation set contains {val_stats['num_ids']} unique suitcase IDs with a total of {val_stats['total_images']} images.\n")
        f.write(f"- There are {val_stats['gallery_count']} gallery images (pre-check) and {val_stats['probe_count']} probe images (post-check).\n")
        f.write(f"- On average, each suitcase ID has {val_stats['avg_gallery_per_id']:.2f} gallery images and {val_stats['avg_probe_per_id']:.2f} probe images.\n\n")
        
        f.write("\n## Visualizations\n\n")
        
        f.write("The following visualizations have been generated to help understand the dataset characteristics:\n\n")
        f.write("1. **Material Distribution**: Distribution of different suitcase materials in the training set\n")
        f.write("2. **Data Source Distribution**: Proportion of pre-check (gallery) vs post-check (probe) images\n")
        f.write("3. **Images per ID**: Distribution showing how many images are available for each suitcase ID\n")
        f.write("4. **View Angle Distribution**: Distribution of different viewing angles in gallery and probe images\n")
        f.write("5. **Image Dimensions**: Scatter plot showing the width and height distribution of images\n")
        f.write("6. **Comparison Plots**: Various comparisons between training and validation sets\n\n")
        
        f.write("All visualizations are saved in the same directory as this report.\n")

    print(f"Analysis complete. Results saved to {output_dir}")
    return report_path

if __name__ == "__main__":
    compare_train_val()
