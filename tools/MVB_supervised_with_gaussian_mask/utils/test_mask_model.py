#!/usr/bin/env python
"""
Test script for ResNet101 with mask generation
"""
import sys
import os
sys.path.insert(0, "/data1/zhaofanghan/SuitcaseReID/OpenUnReID")

import torch
from openunreid.models.backbones.resnet import resnet101_with_mask
from openunreid.models.losses.mask_regression import GaussianMaskRegressionLoss

def test_resnet_with_mask():
    """Test ResNet101 with mask generation"""
    print("Testing ResNet101 with mask generation...")
    
    # Create model
    model = resnet101_with_mask(pretrained=True, mask_output_size=(256, 256))
    model.eval()
    
    # Create dummy input
    batch_size = 2
    inputs = torch.randn(batch_size, 3, 256, 256)
    
    # Forward pass
    with torch.no_grad():
        features, masks = model(inputs)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Masks shape: {masks.shape}")
    
    assert features.shape == (batch_size, 2048, 16, 16), f"Unexpected features shape: {features.shape}"
    assert masks.shape == (batch_size, 1, 256, 256), f"Unexpected masks shape: {masks.shape}"
    
    print("✓ ResNet101 with mask generation test passed!")
    return True

def test_mask_loss():
    """Test Gaussian mask regression loss"""
    print("Testing Gaussian mask regression loss...")
    
    # Create loss function
    loss_fn = GaussianMaskRegressionLoss(loss_type='mse', sigma_ratio=0.1)
    
    # Create dummy data
    batch_size = 2
    pred_masks = torch.sigmoid(torch.randn(batch_size, 1, 256, 256))
    target_bboxes = torch.tensor([
        [0.2, 0.3, 0.8, 0.7],  # bbox for sample 1: [x1, y1, x2, y2]
        [0.1, 0.2, 0.9, 0.8]   # bbox for sample 2
    ])
    
    # Compute loss
    loss = loss_fn(pred_masks, target_bboxes=target_bboxes)
    
    print(f"Predicted masks shape: {pred_masks.shape}")
    print(f"Target bboxes shape: {target_bboxes.shape}")
    print(f"Loss value: {loss.item():.4f}")
    
    assert loss.item() >= 0, "Loss should be non-negative"
    
    print("✓ Gaussian mask regression loss test passed!")
    return True

def test_integration():
    """Test integration of model and loss"""
    print("Testing integration of model and loss...")
    
    # Create model and loss
    model = resnet101_with_mask(pretrained=False, mask_output_size=(256, 256))
    model.train()
    loss_fn = GaussianMaskRegressionLoss(loss_type='mse', sigma_ratio=0.1)
    
    # Create dummy input and target
    batch_size = 2
    inputs = torch.randn(batch_size, 3, 256, 256)
    target_bboxes = torch.tensor([
        [0.2, 0.3, 0.8, 0.7],
        [0.1, 0.2, 0.9, 0.8]
    ])
    
    # Forward pass
    features, masks = model(inputs)
    
    # Compute mask loss
    mask_loss = loss_fn(masks, target_bboxes=target_bboxes)
    
    # Backward pass (to test gradients)
    mask_loss.backward()
    
    print(f"Integration test - mask loss: {mask_loss.item():.4f}")
    
    # Check if gradients are computed
    has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_gradients, "No gradients computed"
    
    print("✓ Integration test passed!")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Testing ResNet101 with Gaussian Mask Regression")
    print("=" * 50)
    
    try:
        test_resnet_with_mask()
        print()
        test_mask_loss()
        print()
        test_integration()
        print()
        print("🎉 All tests passed successfully!")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()