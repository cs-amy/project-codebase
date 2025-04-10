"""
Loss functions for training the deobfuscation model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, Callable
import numpy as np
from skimage.metrics import structural_similarity


class DiceLoss(nn.Module):
    """Dice loss for image segmentation tasks."""
    
    def __init__(self, smooth: float = 1e-6):
        """
        Initialize Dice loss.
        
        Args:
            smooth: Small constant to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to calculate Dice loss.
        
        Args:
            pred: Predicted tensor of shape [batch_size, channels, height, width]
            target: Target tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Dice loss
        """
        # Flatten tensors
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross Entropy and Dice loss."""
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-6):
        """
        Initialize BCE-Dice loss.
        
        Args:
            dice_weight: Weight for Dice loss component
            bce_weight: Weight for BCE loss component
            smooth: Small constant to avoid division by zero
        """
        super(BCEDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to calculate BCE-Dice loss.
        
        Args:
            pred: Predicted tensor of shape [batch_size, channels, height, width]
            target: Target tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Combined BCE-Dice loss
        """
        dice_loss = self.dice_loss(pred, target)
        bce_loss = self.bce_loss(pred, target)
        
        # Combine losses with weights
        combined_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        
        return combined_loss


class FocalLoss(nn.Module):
    """Focal loss for dealing with class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal loss.
        
        Args:
            alpha: Weighting factor for the rare class
            gamma: Focusing parameter (higher values focus more on hard examples)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to calculate Focal loss.
        
        Args:
            pred: Predicted tensor of shape [batch_size, channels, height, width]
            target: Target tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Focal loss
        """
        # Binary focal loss
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)  # pt = p if y=1, pt = 1-p if y=0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) loss."""
    
    def __init__(self, window_size: int = 11, reduction: str = 'mean'):
        """
        Initialize SSIM loss.
        
        Args:
            window_size: Size of the window for SSIM calculation
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.reduction = reduction
        self.channels = 1  # For grayscale images
    
    def _create_window(self, window_size: int, channels: int) -> torch.Tensor:
        """
        Create a Gaussian window for SSIM calculation.
        
        Args:
            window_size: Size of the window
            channels: Number of channels
            
        Returns:
            Gaussian window tensor
        """
        # Create 1D Gaussian window
        sigma = 1.5
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/(2*sigma**2)) 
                             for x in range(window_size)])
        # Normalize
        gauss = gauss / gauss.sum()
        
        # Create 2D Gaussian window
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channels, 1, window_size, window_size).contiguous()
        
        return window
    
    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        Calculate SSIM between two images.
        
        Args:
            img1: First image
            img2: Second image
            window_size: Size of the window
            
        Returns:
            SSIM value
        """
        # Data range (assuming images are normalized to [0, 1])
        data_range = 1.0
        
        # Constants for stability
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        # Create window
        window = self._create_window(window_size, self.channels).to(img1.device)
        
        # Calculate means
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=self.channels)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=self.channels)
        
        # Calculate squares of means
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=self.channels) - mu1_mu2
        
        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Apply reduction
        if self.reduction == 'mean':
            return ssim_map.mean()
        elif self.reduction == 'sum':
            return ssim_map.sum()
        else:  # 'none'
            return ssim_map
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to calculate SSIM loss.
        
        Args:
            pred: Predicted tensor of shape [batch_size, channels, height, width]
            target: Target tensor of shape [batch_size, channels, height, width]
            
        Returns:
            SSIM loss (1 - SSIM)
        """
        ssim_value = self._ssim(pred, target, self.window_size)
        return 1.0 - ssim_value


class L1SSIMLoss(nn.Module):
    """Combined L1 and SSIM loss."""
    
    def __init__(self, ssim_weight: float = 0.5, l1_weight: float = 0.5, window_size: int = 11):
        """
        Initialize L1-SSIM loss.
        
        Args:
            ssim_weight: Weight for SSIM loss component
            l1_weight: Weight for L1 loss component
            window_size: Size of the window for SSIM calculation
        """
        super(L1SSIMLoss, self).__init__()
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.ssim_loss = SSIMLoss(window_size=window_size)
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to calculate L1-SSIM loss.
        
        Args:
            pred: Predicted tensor of shape [batch_size, channels, height, width]
            target: Target tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Combined L1-SSIM loss
        """
        ssim_loss_val = self.ssim_loss(pred, target)
        l1_loss_val = self.l1_loss(pred, target)
        
        # Combine losses with weights
        combined_loss = self.ssim_weight * ssim_loss_val + self.l1_weight * l1_loss_val
        
        return combined_loss


def get_loss_function(loss_name: str) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_name: Name of the loss function
        
    Returns:
        Loss function module
        
    Raises:
        ValueError: If the loss function name is not supported
    """
    loss_functions = {
        'bce': nn.BCELoss(),
        'dice': DiceLoss(),
        'bce_dice': BCEDiceLoss(),
        'focal': FocalLoss(),
        'ssim': SSIMLoss(),
        'l1_ssim': L1SSIMLoss(),
        'mse': nn.MSELoss(),
        'l1': nn.L1Loss()
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unsupported loss function: {loss_name}")
    
    return loss_functions[loss_name] 
