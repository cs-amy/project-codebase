"""
Dataset class for loading letter images.
"""

from pathlib import Path
from typing import Tuple, Optional, List
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LetterDataset(Dataset):
    """Dataset class for loading letter images."""
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[torch.nn.Module] = None,
        target_transform: Optional[torch.nn.Module] = None,
        split: str = 'train'
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            transform: Optional transform to apply to input images
            target_transform: Optional transform to apply to target labels
            split: Dataset split ('train', 'val', or 'test')
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        
        # Validate split
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test']")
        
        # Get list of image files and their labels
        self.image_files, self.labels = self._load_dataset()
        
        # Create label to index mapping
        self.label_to_idx = {chr(i): i - ord('a') for i in range(ord('a'), ord('z') + 1)}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
        logger.info(f"Loaded {len(self.image_files)} images for {split} split")
        logger.info(f"Number of classes: {len(self.label_to_idx)}")
    
    def _load_dataset(self) -> Tuple[List[Path], List[int]]:
        """
        Load dataset files and labels.
        
        Returns:
            Tuple of (list of image file paths, list of label indices)
        """
        image_files = []
        labels = []
        
        # Get all PNG files in the directory
        for img_path in self.root_dir.glob('*.png'):
            # Extract label from filename (assuming format: label_*.png)
            label = img_path.stem.split('_')[0].lower()
            
            # Skip if label is not a single letter
            if len(label) != 1 or not label.isalpha():
                logger.warning(f"Skipping file {img_path} - invalid label format")
                continue
            
            # Convert label to index (a=0, b=1, etc.)
            try:
                label_idx = self.label_to_idx[label]
                image_files.append(img_path)
                labels.append(label_idx)
            except KeyError:
                logger.warning(f"Skipping file {img_path} - invalid label: {label}")
                continue
        
        return image_files, labels
    
    def __len__(self) -> int:
        """Get the total number of samples in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Tuple of (image tensor, label tensor)
        """
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Get label
        label = self.labels[idx]
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def get_class_counts(self) -> dict:
        """
        Get the count of samples for each class.
        
        Returns:
            Dictionary mapping class indices to their counts
        """
        counts = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling class imbalance.
        
        Returns:
            Tensor of class weights
        """
        counts = self.get_class_counts()
        total = len(self)
        weights = torch.zeros(len(self.label_to_idx))
        
        for idx, count in counts.items():
            weights[idx] = total / (len(counts) * count)
        
        return weights 