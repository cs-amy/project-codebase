"""
Data loader module for loading and processing image pairs for model training.
"""

import os
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from pathlib import Path
import tensorflow as tf # type: ignore
import torch
from torch.utils.data import Dataset, DataLoader
import logging

from src.utils.image_processing import (
    resize_image,
    normalize_image,
    random_rotation,
    add_gaussian_noise
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CharacterImageDataset(Dataset):
    """PyTorch Dataset for character image pairs (obfuscated -> standard)."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (64, 64),
        augment: bool = True,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory of the dataset
            split: Data split ('train', 'val', or 'test')
            image_size: Size of the images as (height, width)
            augment: Whether to apply augmentation (only for training)
            transform: Transform to apply to the input (obfuscated) images
            target_transform: Transform to apply to the target (standard) images
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and split == 'train'  # Only augment training data
        self.transform = transform
        self.target_transform = target_transform
        
        # Get paths to images - updated to work with our directory structure
        self.obfuscated_dir = self.data_dir / "obfuscated" / split
        self.standard_dir = self.data_dir / "regular" / split
        
        if not self.obfuscated_dir.exists():
            raise ValueError(f"Obfuscated directory not found: {self.obfuscated_dir}")
        if not self.standard_dir.exists():
            raise ValueError(f"Regular directory not found: {self.standard_dir}")
        
        # Get list of image files - finding character directories
        self.obfuscated_chars = sorted([d for d in os.listdir(self.obfuscated_dir) if os.path.isdir(os.path.join(self.obfuscated_dir, d))])
        self.standard_chars = sorted([d for d in os.listdir(self.standard_dir) if os.path.isdir(os.path.join(self.standard_dir, d))])
        
        # Ensure we have matching characters
        self.chars = sorted(list(set(self.obfuscated_chars).intersection(set(self.standard_chars))))
        if not self.chars:
            raise ValueError(f"No matching character directories found for split '{split}'")
        
        # Load all image pairs
        self.image_pairs = []
        
        for char in self.chars:
            obfuscated_char_dir = os.path.join(self.obfuscated_dir, char)
            standard_char_dir = os.path.join(self.standard_dir, char)
            
            # Get all obfuscated images
            obfuscated_images = [f for f in os.listdir(obfuscated_char_dir) if f.endswith('.png')]
            standard_images = [f for f in os.listdir(standard_char_dir) if f.endswith('.png')]
            
            # Pair each obfuscated image with a random standard image of the same character
            for obf_img in obfuscated_images:
                if standard_images:
                    # Pick a random standard image of the same character
                    std_img = np.random.choice(standard_images)
                    self.image_pairs.append({
                        'obfuscated': os.path.join(obfuscated_char_dir, obf_img),
                        'standard': os.path.join(standard_char_dir, std_img),
                        'char': char
                    })
        
        if len(self.image_pairs) == 0:
            raise ValueError(f"No image pairs found for split '{split}'")
        
        logger.info(f"Loaded {len(self.image_pairs)} image pairs for '{split}' split across {len(self.chars)} characters")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (obfuscated_image, standard_image) as torch tensors
        """
        pair = self.image_pairs[idx]
        
        # Load images
        obfuscated_path = pair['obfuscated']
        standard_path = pair['standard']
        
        obfuscated_img = cv2.imread(obfuscated_path, cv2.IMREAD_GRAYSCALE)
        standard_img = cv2.imread(standard_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize if needed
        if obfuscated_img.shape[:2] != self.image_size:
            obfuscated_img = resize_image(obfuscated_img, self.image_size)
        if standard_img.shape[:2] != self.image_size:
            standard_img = resize_image(standard_img, self.image_size)
        
        # Normalize to [0, 1]
        obfuscated_img = normalize_image(obfuscated_img)
        standard_img = normalize_image(standard_img)
        
        # Apply augmentation if needed
        if self.augment:
            if np.random.random() < 0.5:
                obfuscated_img = random_rotation(obfuscated_img, max_angle=10)
            if np.random.random() < 0.3:
                obfuscated_img = add_gaussian_noise(obfuscated_img, mean=0, std=0.02)
        
        # Apply transformations if provided
        if self.transform is not None:
            obfuscated_img = self.transform(obfuscated_img)
        if self.target_transform is not None:
            standard_img = self.target_transform(standard_img)
        
        # Convert to PyTorch tensors
        obfuscated_tensor = torch.tensor(obfuscated_img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        standard_tensor = torch.tensor(standard_img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        return obfuscated_tensor, standard_tensor


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (64, 64),
    num_workers: int = 4,
    augment: bool = True
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Root directory of the dataset
        batch_size: Batch size for the data loaders
        image_size: Size of the images as (height, width)
        num_workers: Number of worker processes for data loading
        augment: Whether to apply augmentation to training data
        
    Returns:
        Dictionary of data loaders for each split
    """
    loaders = {}
    
    # Create train loader
    train_dataset = CharacterImageDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size,
        augment=augment
    )
    
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Use test data for both validation and testing since we don't have a separate val split
    test_dataset = CharacterImageDataset(
        data_dir=data_dir,
        split='test',
        image_size=image_size,
        augment=False
    )
    
    # Create val loader - using a subset of the test data (50%)
    val_indices = torch.randperm(len(test_dataset))[:len(test_dataset) // 2]
    test_indices = torch.randperm(len(test_dataset))[len(test_dataset) // 2:]
    
    val_dataset = torch.utils.data.Subset(test_dataset, val_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    loaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create test loader
    loaders['test'] = DataLoader(
        test_subset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders: train={len(loaders['train'].dataset)} samples, val={len(loaders['val'].dataset)} samples, test={len(loaders['test'].dataset)} samples")
    
    return loaders


class TensorFlowDataLoader:
    """TensorFlow data loader for character image pairs."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        image_size: Tuple[int, int] = (64, 64),
        augment: bool = True,
        buffer_size: int = 1000
    ):
        """
        Initialize the TensorFlow data loader.
        
        Args:
            data_dir: Root directory of the dataset
            batch_size: Batch size for the data loaders
            image_size: Size of the images as (height, width)
            augment: Whether to apply augmentation to training data
            buffer_size: Buffer size for shuffling
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.buffer_size = buffer_size
        
        # Create TensorFlow datasets
        self.datasets = {}
        for split in ['train', 'val', 'test']:
            self.datasets[split] = self._create_tf_dataset(split)
    
    def _parse_image(self, filename: str, split: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Parse image files and create input-output pairs.
        
        Args:
            filename: Image filename
            split: Data split ('train', 'val', or 'test')
            
        Returns:
            Tuple of (obfuscated_image, standard_image) as TensorFlow tensors
        """
        obfuscated_path = tf.strings.join([self.data_dir / split / 'obfuscated', filename], '/')
        standard_path = tf.strings.join([self.data_dir / split / 'standard', filename], '/')
        
        # Read images
        obfuscated_img = tf.io.read_file(obfuscated_path)
        standard_img = tf.io.read_file(standard_path)
        
        # Decode images
        obfuscated_img = tf.image.decode_png(obfuscated_img, channels=1)
        standard_img = tf.image.decode_png(standard_img, channels=1)
        
        # Resize if needed
        obfuscated_img = tf.image.resize(obfuscated_img, self.image_size)
        standard_img = tf.image.resize(standard_img, self.image_size)
        
        # Normalize to [0, 1]
        obfuscated_img = tf.cast(obfuscated_img, tf.float32) / 255.0
        standard_img = tf.cast(standard_img, tf.float32) / 255.0
        
        return obfuscated_img, standard_img
    
    def _augment(self, obfuscated_img: tf.Tensor, standard_img: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply augmentations to the images.
        
        Args:
            obfuscated_img: Input obfuscated image
            standard_img: Target standard image
            
        Returns:
            Tuple of augmented (obfuscated_image, standard_image)
        """
        # Random rotation
        if tf.random.uniform([], minval=0, maxval=1) < 0.5:
            angle = tf.random.uniform([], minval=-10, maxval=10) * np.pi / 180.0
            obfuscated_img = tf.keras.layers.experimental.preprocessing.RandomRotation(
                factor=angle)(obfuscated_img)
        
        # Random noise
        if tf.random.uniform([], minval=0, maxval=1) < 0.3:
            noise = tf.random.normal(tf.shape(obfuscated_img), mean=0.0, stddev=0.02)
            obfuscated_img = tf.clip_by_value(obfuscated_img + noise, 0.0, 1.0)
        
        return obfuscated_img, standard_img
    
    def _create_tf_dataset(self, split: str) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset for a given split.
        
        Args:
            split: Data split ('train', 'val', or 'test')
            
        Returns:
            TensorFlow dataset
        """
        split_dir = self.data_dir / split
        obfuscated_dir = split_dir / 'obfuscated'
        
        # Get list of image files
        filenames = tf.data.Dataset.list_files(str(obfuscated_dir / '*.png'))
        
        # Extract just the filename without the path
        filenames = filenames.map(lambda x: tf.strings.split(x, '/')[-1])
        
        # Parse images
        dataset = filenames.map(
            lambda x: self._parse_image(x, split),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply augmentation only to training data
        if self.augment and split == 'train':
            dataset = dataset.map(
                self._augment,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Batch and prefetch
        dataset = dataset.shuffle(self.buffer_size) if split == 'train' else dataset
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_dataset(self, split: str) -> tf.data.Dataset:
        """
        Get the TensorFlow dataset for a specific split.
        
        Args:
            split: Data split ('train', 'val', or 'test')
            
        Returns:
            TensorFlow dataset
        """
        if split not in self.datasets:
            raise ValueError(f"Invalid split '{split}'. Must be one of: train, val, test")
        
        return self.datasets[split] 