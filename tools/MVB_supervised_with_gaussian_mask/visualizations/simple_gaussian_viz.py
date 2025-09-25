#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Gaussian Mask Visualization (English version)
"""

import sys
import os

# 添加正确的路径
sys.path.insert(0, "/data1/zhaofanghan/SuitcaseReID/OpenUnReID")
os.chdir("/data1/zhaofanghan/SuitcaseReID/OpenUnReID")

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 直接导入mask_regression模块中的类
try:
    from openunreid.models.losses.mask_regression import GaussianMaskRegressionLoss
except ImportError as e:
    print(f"Import error: {e}")
    print("Loading mask_regression.py file directly...")
    
    # 如果模块导入失败，直接加载文件
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mask_regression", 
        "/data1/zhaofanghan/SuitcaseReID/OpenUnReID/openunreid/models/losses/mask_regression.py"
    )
    mask_regression = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mask_regression)
    GaussianMaskRegressionLoss = mask_regression.GaussianMaskRegressionLoss


def create_simple_visualization():
    """
    Create a simple visualization showing Gaussian mask center values
    """
    mask_loss = GaussianMaskRegressionLoss(sigma_ratio=0.1)
    
    # Generate different Gaussian masks
    height, width = 64, 64
    
    # Test cases with different parameters
    test_cases = [
        {'center_x': 0.5, 'center_y': 0.5, 'sigma_x': 0.05, 'sigma_y': 0.05, 'name': 'Small sigma'},
        {'center_x': 0.5, 'center_y': 0.5, 'sigma_x': 0.1, 'sigma_y': 0.1, 'name': 'Medium sigma'},
        {'center_x': 0.5, 'center_y': 0.5, 'sigma_x': 0.15, 'sigma_y': 0.15, 'name': 'Large sigma'},
        {'center_x': 0.3, 'center_y': 0.3, 'sigma_x': 0.1, 'sigma_y': 0.1, 'name': 'Off-center'},
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Gaussian Mask Visualization - Center Value = 1.0', fontsize=16, fontweight='bold')
    
    for i, case in enumerate(test_cases):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Generate Gaussian mask
        mask = mask_loss.generate_gaussian_mask(
            height, width,
            case['center_x'], case['center_y'],
            case['sigma_x'], case['sigma_y']
        )
        
        mask_np = mask.detach().cpu().numpy()
        
        # Create heatmap
        im = ax.imshow(mask_np, cmap='hot', interpolation='bilinear', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Gaussian Value', rotation=270, labelpad=15)
        
        # Mark center point
        center_x_pixel = case['center_x'] * width
        center_y_pixel = case['center_y'] * height
        ax.plot(center_x_pixel, center_y_pixel, 'w+', markersize=15, markeredgewidth=3)
        ax.plot(center_x_pixel, center_y_pixel, 'k+', markersize=12, markeredgewidth=2)
        
        # Set title and labels
        ax.set_title(f'{case["name"]}\nCenter=({case["center_x"]:.1f}, {case["center_y"]:.1f}), '
                    f'Sigma=({case["sigma_x"]:.2f}, {case["sigma_y"]:.2f})', 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('X coordinate (pixels)')
        ax.set_ylabel('Y coordinate (pixels)')
        
        # Show center value
        center_val = mask_np[int(center_y_pixel), int(center_x_pixel)]
        max_val = mask_np.max()
        ax.text(0.02, 0.98, f'Max: {max_val:.3f}\nCenter: {center_val:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/data1/zhaofanghan/SuitcaseReID/OpenUnReID/tools/MVB_supervised_with_gaussian_mask/simple_gaussian_viz.png', 
                dpi=300, bbox_inches='tight')
    print("✅ Simple Gaussian visualization saved: simple_gaussian_viz.png")
    plt.show()


def create_cross_section_plot():
    """
    Create cross-section plot to show Gaussian center value clearly
    """
    mask_loss = GaussianMaskRegressionLoss(sigma_ratio=0.1)
    
    # Generate a standard Gaussian mask
    height, width = 128, 128
    center_x, center_y = 0.5, 0.5
    sigma_x, sigma_y = 0.1, 0.1
    
    mask = mask_loss.generate_gaussian_mask(height, width, center_x, center_y, sigma_x, sigma_y)
    mask_np = mask.detach().cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Gaussian Mask Cross-Section Analysis', fontsize=16, fontweight='bold')
    
    # Left: Heatmap with cross-section lines
    im = ax1.imshow(mask_np, cmap='hot', interpolation='bilinear', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='Gaussian Value')
    
    center_x_pixel = int(center_x * width)
    center_y_pixel = int(center_y * height)
    
    # Draw cross-section lines
    ax1.axhline(y=center_y_pixel, color='cyan', linestyle='-', linewidth=2, alpha=0.8, label='Horizontal cut')
    ax1.axvline(x=center_x_pixel, color='lime', linestyle='-', linewidth=2, alpha=0.8, label='Vertical cut')
    ax1.plot(center_x_pixel, center_y_pixel, 'w+', markersize=20, markeredgewidth=4)
    ax1.plot(center_x_pixel, center_y_pixel, 'k+', markersize=16, markeredgewidth=3)
    
    ax1.set_title('2D Heatmap with Cross-sections')
    ax1.set_xlabel('X coordinate (pixels)')
    ax1.set_ylabel('Y coordinate (pixels)')
    ax1.legend()
    
    # Right: Cross-section plot
    horizontal_section = mask_np[center_y_pixel, :]
    x_coords = np.arange(width)
    
    ax2.plot(x_coords, horizontal_section, 'cyan', linewidth=3, label='Horizontal cross-section')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Theoretical max = 1.0')
    ax2.axvline(x=center_x_pixel, color='red', linestyle='--', alpha=0.7, label=f'Center pixel = {center_x_pixel}')
    
    # Mark the center value
    center_value = horizontal_section[center_x_pixel]
    ax2.plot(center_x_pixel, center_value, 'ro', markersize=10, label=f'Center value = {center_value:.4f}')
    
    ax2.set_title('Cross-section through Center')
    ax2.set_xlabel('X coordinate (pixels)')
    ax2.set_ylabel('Gaussian Value')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('/data1/zhaofanghan/SuitcaseReID/OpenUnReID/tools/MVB_supervised_with_gaussian_mask/cross_section_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("✅ Cross-section analysis saved: cross_section_analysis.png")
    plt.show()


def print_center_value_analysis():
    """
    Print detailed analysis of Gaussian center values
    """
    print("\n" + "="*60)
    print("GAUSSIAN MASK CENTER VALUE ANALYSIS")
    print("="*60)
    
    mask_loss = GaussianMaskRegressionLoss(sigma_ratio=0.1)
    
    # Test different resolutions and sigma values
    test_configs = [
        {'size': (32, 32), 'sigma': 0.1},
        {'size': (64, 64), 'sigma': 0.1},
        {'size': (128, 128), 'sigma': 0.1},
        {'size': (256, 256), 'sigma': 0.1},
        {'size': (64, 64), 'sigma': 0.05},
        {'size': (64, 64), 'sigma': 0.15},
    ]
    
    for config in test_configs:
        height, width = config['size']
        sigma = config['sigma']
        
        mask = mask_loss.generate_gaussian_mask(height, width, 0.5, 0.5, sigma, sigma)
        mask_np = mask.detach().cpu().numpy()
        
        center_pixel_x = int(0.5 * width)
        center_pixel_y = int(0.5 * height)
        center_value = mask_np[center_pixel_y, center_pixel_x]
        max_value = mask_np.max()
        
        print(f"Size: {height}x{width}, Sigma: {sigma:.2f}")
        print(f"  Center pixel: ({center_pixel_x}, {center_pixel_y})")
        print(f"  Center value: {center_value:.6f}")
        print(f"  Max value: {max_value:.6f}")
        print(f"  Difference from 1.0: {1.0 - center_value:.6f}")
        print()
    
    print("KEY FINDINGS:")
    print("1. Theoretical Gaussian center value = exp(0) = 1.0")
    print("2. Actual center value approaches 1.0 but may be slightly less due to:")
    print("   - Discrete pixel sampling")
    print("   - Floating point precision")
    print("   - Grid alignment effects")
    print("3. Higher resolutions give values closer to 1.0")
    print("4. Center value is independent of sigma (always ≈ 1.0)")
    print("="*60)


def main():
    """
    Main function to run all visualizations
    """
    print("Creating Gaussian mask visualizations...")
    
    # Set matplotlib to use default fonts (avoid Chinese font issues)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    try:
        # 1. Simple visualization
        print("\n1. Creating simple Gaussian mask visualization...")
        create_simple_visualization()
        
        # 2. Cross-section analysis
        print("\n2. Creating cross-section analysis...")
        create_cross_section_plot()
        
        # 3. Detailed analysis
        print_center_value_analysis()
        
        output_dir = "/data1/zhaofanghan/SuitcaseReID/OpenUnReID/tools/MVB_supervised_with_gaussian_mask"
        print(f"\n✅ All visualizations saved to: {output_dir}")
        print("📊 Generated files:")
        print("  - simple_gaussian_viz.png: Basic Gaussian mask examples")
        print("  - cross_section_analysis.png: Cross-section through Gaussian center")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()