"""
CNN model for letter classification.
"""

import torch.nn as nn
import torch.nn.functional as F

class LetterClassifierCNN(nn.Module):
    """
    CNN model for classifying letters in images.
    Input: Single channel image (grayscale)
    Output: 26-class prediction (a-z)
    """
    
    def __init__(self, input_shape=(64, 64), num_classes=26):
        """
        Initialize the model.
        
        Args:
            input_shape: Tuple of (height, width) for input images
            num_classes: Number of classes (default: 26 for a-z)
        """
        super(LetterClassifierCNN, self).__init__()
        
        # Calculate the size of the flattened features after convolutions
        self.input_shape = input_shape
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size of flattened features
        # After 3 pooling layers, the size is reduced by 2^3 = 8
        self.flat_size = 128 * (input_shape[0] // 8) * (input_shape[1] // 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First convolutional block
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second convolutional block
        x = self.pool(F.relu(self.conv2(x)))
        
        # Third convolutional block
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, self.flat_size)
        
        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

def get_model(input_shape=(64, 64), num_classes=26):
    """
    Factory function to create the model.
    
    Args:
        input_shape: Tuple of (height, width) for input images
        num_classes: Number of classes (default: 26 for a-z)
        
    Returns:
        Initialized model
    """
    return LetterClassifierCNN(input_shape=input_shape, num_classes=num_classes) 
