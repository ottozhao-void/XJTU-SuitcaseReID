#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for Gaussian mask regression loss with threshold control
"""

import sys
import os
sys.path.insert(0, "/data1/zhaofanghan/SuitcaseReID/OpenUnReID")

import torch
import numpy as np
import matplotlib.pyplot as plt

from openunreid.models.losses.mask_regression import GaussianMaskRegressionLoss


def test_threshold_functionality():
    """
    Test the threshold functionality in GaussianMaskRegressionLoss
    """
    print("="*60)
    print("Testing Gaussian Mask Regression Loss with Threshold Control")
    print("="*60)
    
    # Create loss functions with different thresholds
    loss_no_threshold = GaussianMaskRegressionLoss(
        loss_type='mse', 
        sigma_ratio=0.1, 
        weight=1.0, 
        threshold=0.0
    )
    
    loss_low_threshold = GaussianMaskRegressionLoss(
        loss_type='mse', 
        sigma_ratio=0.1, 
        weight=1.0, 
        threshold=0.1
    )
    
    loss_high_threshold = GaussianMaskRegressionLoss(
        loss_type='mse', 
        sigma_ratio=0.1, 
        weight=1.0, 
        threshold=0.5
    )
    
    # Create test data
    batch_size = 2
    height, width = 64, 64
    
    # Create predicted masks (random values)
    pred_masks = torch.sigmoid(torch.randn(batch_size, 1, height, width))
    
    # Create target bounding boxes
    target_bboxes = torch.tensor([
        [0.2, 0.3, 0.8, 0.7],  # bbox for sample 1
        [0.3, 0.2, 0.7, 0.8]   # bbox for sample 2
    ], dtype=torch.float32)
    
    # Compute losses with different thresholds
    loss_0 = loss_no_threshold(pred_masks, target_bboxes=target_bboxes)
    loss_1 = loss_low_threshold(pred_masks, target_bboxes=target_bboxes)
    loss_5 = loss_high_threshold(pred_masks, target_bboxes=target_bboxes)
    
    print(f"Loss with threshold = 0.0:  {loss_0.item():.6f}")
    print(f"Loss with threshold = 0.1:  {loss_1.item():.6f}")
    print(f"Loss with threshold = 0.5:  {loss_5.item():.6f}")
    
    print("\nAnalysis:")
    print("- As threshold increases, fewer pixels contribute to the loss")
    print("- Higher threshold focuses on high-confidence regions (center of Gaussian)")
    print("- Lower threshold includes more peripheral regions")
    
    return loss_0, loss_1, loss_5


def visualize_threshold_effect():
    """
    Visualize the effect of different thresholds on mask processing
    """
    print("\n" + "="*60)
    print("Visualizing Threshold Effects on Gaussian Masks")
    print("="*60)
    
    # Create loss function
    mask_loss = GaussianMaskRegressionLoss(sigma_ratio=0.1)
    
    # Generate a target Gaussian mask
    height, width = 128, 128
    center_x, center_y = 0.5, 0.5
    sigma_x, sigma_y = 0.1, 0.1
    
    target_mask = mask_loss.generate_gaussian_mask(
        height, width, center_x, center_y, sigma_x, sigma_y
    )
    target_mask = target_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    # Define different thresholds
    thresholds = [0.0, 0.1, 0.3, 0.5, 0.7]
    
    # Create figure
    fig, axes = plt.subplots(2, len(thresholds), figsize=(20, 8))
    fig.suptitle('Effect of Different Thresholds on Gaussian Mask Loss', fontsize=16, fontweight='bold')
    
    for i, threshold in enumerate(thresholds):
        # Create threshold mask
        threshold_mask = (target_mask >= threshold).float()
        
        # Original mask
        axes[0, i].imshow(target_mask[0, 0].detach().numpy(), cmap='hot', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original Mask')
        axes[0, i].axis('off')
        
        # Threshold mask
        im = axes[1, i].imshow(threshold_mask[0, 0].detach().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Threshold = {threshold}')
        axes[1, i].axis('off')
        
        # Add text showing percentage of pixels above threshold
        valid_pixels = threshold_mask.sum().item()
        total_pixels = threshold_mask.numel()
        percentage = (valid_pixels / total_pixels) * 100
        
        axes[1, i].text(0.5, -0.1, f'{percentage:.1f}% pixels', 
                       transform=axes[1, i].transAxes, ha='center', va='top')
    
    plt.tight_layout()
    plt.savefig('/data1/zhaofanghan/SuitcaseReID/OpenUnReID/tools/MVB_supervised_with_gaussian_mask/threshold_effect_visualization.png', 
                dpi=300, bbox_inches='tight')
    print("✅ Threshold effect visualization saved: threshold_effect_visualization.png")
    
    # Show statistics
    print("\nThreshold Statistics:")
    for threshold in thresholds:
        threshold_mask = (target_mask >= threshold).float()
        valid_pixels = threshold_mask.sum().item()
        total_pixels = threshold_mask.numel()
        percentage = (valid_pixels / total_pixels) * 100
        print(f"Threshold {threshold}: {valid_pixels:>5} pixels ({percentage:>5.1f}%) contribute to loss")


def compare_loss_computation_methods():
    """
    Compare loss computation with and without threshold
    """
    print("\n" + "="*60)
    print("Comparing Loss Computation Methods")
    print("="*60)
    
    # Create test data
    batch_size = 1
    height, width = 32, 32
    
    # Create a simple predicted mask (uniform values)
    pred_masks = torch.ones(batch_size, 1, height, width) * 0.3
    
    # Create target mask (Gaussian)
    loss_fn = GaussianMaskRegressionLoss(sigma_ratio=0.1)
    target_bboxes = torch.tensor([[0.25, 0.25, 0.75, 0.75]], dtype=torch.float32)
    target_masks = loss_fn.create_gaussian_masks_from_bbox(
        target_bboxes, 
        mask_size=(height, width)
    )
    
    print(f"Prediction mask shape: {pred_masks.shape}")
    print(f"Target mask shape: {target_masks.shape}")
    print(f"Predicted mask values (uniform): {pred_masks[0,0,0,0].item():.3f}")
    print(f"Target mask center value: {target_masks[0,0,height//2,width//2].item():.6f}")
    print(f"Target mask corner value: {target_masks[0,0,0,0].item():.6f}")
    
    # Test different thresholds
    thresholds = [0.0, 0.1, 0.3, 0.5, 0.8]
    
    print(f"\nLoss comparison (MSE):")
    for threshold in thresholds:
        loss_fn_thresh = GaussianMaskRegressionLoss(
            loss_type='mse', 
            sigma_ratio=0.1, 
            weight=1.0, 
            threshold=threshold
        )
        
        loss = loss_fn_thresh(pred_masks, target_masks=target_masks)
        
        # Calculate number of contributing pixels
        threshold_mask = (target_masks >= threshold).float()
        contributing_pixels = threshold_mask.sum().item()
        total_pixels = target_masks.numel()
        
        print(f"Threshold {threshold}: Loss = {loss.item():.6f}, "
              f"Contributing pixels = {contributing_pixels:>4} ({contributing_pixels/total_pixels*100:>5.1f}%)")


def main():
    """
    Main test function
    """
    try:
        # Test basic threshold functionality
        test_threshold_functionality()
        
        # Visualize threshold effects
        visualize_threshold_effect()
        
        # Compare loss computation methods
        compare_loss_computation_methods()
        
        print("\n🎉 All threshold tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()