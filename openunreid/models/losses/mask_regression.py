# Written for Gaussian Mask Regression Loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianMaskRegressionLoss(nn.Module):
    """
    Gaussian Mask Regression Loss for suitcase ReID with attention masks.
    
    This loss computes the regression loss between predicted masks and 
    ground truth Gaussian masks.
    """
    
    def __init__(self, loss_type='mse', sigma_ratio=0.1, weight=1.0, threshold=0.0):
        super(GaussianMaskRegressionLoss, self).__init__()
        self.loss_type = loss_type
        self.sigma_ratio = sigma_ratio  # Gaussian sigma as ratio of image size
        self.weight = weight
        self.threshold = threshold  # Threshold for pixel contribution to loss
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='none')  # Change to 'none' for element-wise loss
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def generate_gaussian_mask(self, height, width, center_x, center_y, sigma_x, sigma_y, device=None):
        """
        Generate a 2D Gaussian mask.
        
        Args:
            height, width: mask dimensions
            center_x, center_y: center of the Gaussian (normalized coordinates [0,1])
            sigma_x, sigma_y: standard deviations
            device: torch device to create tensors on
        
        Returns:
            Gaussian mask of shape (height, width)
        """
        # Ensure we use the right device
        if device is None:
            if torch.is_tensor(center_x):
                device = center_x.device
            else:
                device = torch.device('cpu')
        
        # Create coordinate grids on the correct device
        y_coords = torch.linspace(0, 1, height, device=device).unsqueeze(1).expand(height, width)
        x_coords = torch.linspace(0, 1, width, device=device).unsqueeze(0).expand(height, width)
        
        # Convert scalar values to tensors on the correct device if needed
        if not torch.is_tensor(center_x):
            center_x = torch.tensor(center_x, device=device)
        if not torch.is_tensor(center_y):
            center_y = torch.tensor(center_y, device=device)
        if not torch.is_tensor(sigma_x):
            sigma_x = torch.tensor(sigma_x, device=device)
        if not torch.is_tensor(sigma_y):
            sigma_y = torch.tensor(sigma_y, device=device)
        
        # Calculate Gaussian
        gaussian = torch.exp(-0.5 * (
            ((x_coords - center_x) / sigma_x) ** 2 + 
            ((y_coords - center_y) / sigma_y) ** 2
        ))
        
        return gaussian
    
    def create_gaussian_masks_from_bbox(self, bboxes, mask_size=(256, 256)):
        """
        Create Gaussian masks from bounding box annotations.
        
        Args:
            bboxes: Tensor of shape (batch_size, 4) with normalized bbox coordinates [x1, y1, x2, y2]
            mask_size: Target mask size (height, width)
        
        Returns:
            Gaussian masks of shape (batch_size, 1, height, width)
        """
        batch_size = bboxes.shape[0]
        height, width = mask_size
        masks = torch.zeros(batch_size, 1, height, width, device=bboxes.device)
        
        for i in range(batch_size):
            x1, y1, x2, y2 = bboxes[i]
            
            # Calculate center and size
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Calculate sigma based on bbox size and ratio
            sigma_x = bbox_width * self.sigma_ratio
            sigma_y = bbox_height * self.sigma_ratio
            
            # Generate Gaussian mask
            gaussian_mask = self.generate_gaussian_mask(
                height, width, center_x, center_y, sigma_x, sigma_y, device=bboxes.device
            )
            
            masks[i, 0] = gaussian_mask
        
        return masks
    
    def forward(self, pred_masks, target_bboxes=None, target_masks=None):
        """
        Compute mask regression loss with threshold control.
        
        Args:
            pred_masks: Predicted masks of shape (batch_size, 1, height, width)
            target_bboxes: Target bounding boxes of shape (batch_size, 4) [x1, y1, x2, y2]
            target_masks: Pre-computed target masks of shape (batch_size, 1, height, width)
        
        Returns:
            Loss value
        """
        if target_masks is None:
            if target_bboxes is None:
                raise ValueError("Either target_bboxes or target_masks must be provided")
            target_masks = self.create_gaussian_masks_from_bbox(
                target_bboxes, 
                mask_size=(pred_masks.shape[2], pred_masks.shape[3])
            )
        
        # Ensure masks are in the same device
        target_masks = target_masks.to(pred_masks.device)
        
        # Compute element-wise regression loss
        element_loss = self.criterion(pred_masks, target_masks)
        
        # Apply threshold mask - only pixels above threshold contribute to loss
        if self.threshold > 0.0:
            # Create threshold mask based on target mask values
            threshold_mask = (target_masks >= self.threshold).float()
            
            # Apply threshold mask to element-wise loss
            masked_loss = element_loss * threshold_mask
            
            # Compute mean only over valid (above threshold) pixels
            num_valid_pixels = threshold_mask.sum()
            if num_valid_pixels > 0:
                loss = masked_loss.sum() / num_valid_pixels
            else:
                # If no pixels above threshold, use regular mean
                loss = element_loss.mean()
        else:
            # No threshold, use regular mean
            loss = element_loss.mean()
        
        return self.weight * loss


class FocalMaskLoss(nn.Module):
    """
    Focal loss variant for mask regression to handle imbalanced regions.
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, weight=1.0):
        super(FocalMaskLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, pred_masks, target_masks):
        """
        Compute focal mask loss.
        
        Args:
            pred_masks: Predicted masks of shape (batch_size, 1, height, width)
            target_masks: Target masks of shape (batch_size, 1, height, width)
        
        Returns:
            Loss value
        """
        # Compute BCE loss element-wise
        bce_loss = F.binary_cross_entropy(pred_masks, target_masks, reduction='none')
        
        # Compute pt
        pt = torch.where(target_masks == 1, pred_masks, 1 - pred_masks)
        
        # Compute focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight to BCE loss
        focal_loss = focal_weight * bce_loss
        
        return self.weight * focal_loss.mean()


class CombinedMaskLoss(nn.Module):
    """
    Combined loss for mask regression including both regression and focal loss.
    """
    
    def __init__(self, 
                 regression_weight=1.0, 
                 focal_weight=0.5,
                 regression_type='mse',
                 sigma_ratio=0.1):
        super(CombinedMaskLoss, self).__init__()
        
        self.regression_loss = GaussianMaskRegressionLoss(
            loss_type=regression_type,
            sigma_ratio=sigma_ratio,
            weight=regression_weight
        )
        
        self.focal_loss = FocalMaskLoss(weight=focal_weight)
    
    def forward(self, pred_masks, target_bboxes=None, target_masks=None):
        """
        Compute combined mask loss.
        
        Args:
            pred_masks: Predicted masks of shape (batch_size, 1, height, width)
            target_bboxes: Target bounding boxes of shape (batch_size, 4)
            target_masks: Pre-computed target masks of shape (batch_size, 1, height, width)
        
        Returns:
            Combined loss value
        """
        # Regression loss
        reg_loss = self.regression_loss(pred_masks, target_bboxes, target_masks)
        
        # Generate target masks if not provided
        if target_masks is None:
            target_masks = self.regression_loss.create_gaussian_masks_from_bbox(
                target_bboxes, 
                mask_size=(pred_masks.shape[2], pred_masks.shape[3])
            )
        
        # Focal loss
        focal_loss_val = self.focal_loss(pred_masks, target_masks)
        
        total_loss = reg_loss + focal_loss_val
        
        return total_loss