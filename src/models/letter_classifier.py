"""
CNN model architecture for character deobfuscation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class LetterClassifierCNN(nn.Module):
    """
    CNN architecture for deobfuscating characters.
    
    Architecture overview:
    1. Feature Extraction: Multiple convolutional layers with increasing channels
    2. Spatial Reduction: Max pooling layers to reduce dimensionality
    3. Regularization: Dropout and batch normalization to prevent overfitting
    4. Classification: Fully connected layers for final character prediction
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (28, 28, 1),
        num_classes: int = 26,
        dropout_rate: float = 0.5
    ):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Tuple of (height, width, channels)
            num_classes: Number of output classes (26 for a-z)
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError("Input shape must be (height, width, channels)")
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block: 32 filters, 3x3 kernel
            nn.Conv2d(input_shape[2], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate/2),
            
            # Second conv block: 64 filters, 3x3 kernel
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate/2),
            
            # Third conv block: 128 filters, 3x3 kernel
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate)
        )
        
        # Calculate size of flattened features
        # After 3 max pooling layers with stride 2: input_dim / 8
        feature_size = (input_shape[0] // 8) * (input_shape[1] // 8) * 128
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Tensor of shape (batch_size, num_classes)
        """
        # Feature extraction
        x = self.features(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x

def get_model(architecture: str, config: Dict) -> nn.Module:
    """
    Factory function to create a model instance.
    
    Args:
        architecture: Name of the model architecture
        config: Model configuration dictionary
    
    Returns:
        Instantiated model
    """
    if architecture == "LetterClassifierCNN":
        return LetterClassifierCNN(
            input_shape=config.get("input_shape", (28, 28, 1)),
            num_classes=config.get("num_classes", 26),
            dropout_rate=config.get("dropout_rate", 0.5)
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}") 
