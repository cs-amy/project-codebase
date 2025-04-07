"""
Data loader module for loading and processing image pairs for model training.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image

# TensorFlow imports (only used in TensorFlowDataLoader)
import tensorflow as tf # type: ignore
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_data_directory(data_dir: Path) -> None:
    """
    Validate the data directory structure.

    Args:
        data_dir: Path to the data/characters directory
    """
    required_dirs = [
        "regular/train",
        "regular/test",
        "obfuscated/train",
        "obfuscated/test"
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        full_path = data_dir / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)

    if missing_dirs:
        raise FileNotFoundError(
            f"Missing required directories in {data_dir}:\n" +
            "\n".join(f"- {d}" for d in missing_dirs)
        )

    # Log directory structure
    logger.info(f"Data directory structure validated:")
    logger.info(f"Root: {data_dir}")
    for dir_path in required_dirs:
        full_path = data_dir / dir_path
        num_files = len(list(full_path.glob("**/*.png")))
        logger.info(f"- {dir_path}: {num_files} PNG files")


class CharacterDataset(Dataset):
    """Dataset for character images (regular or obfuscated)."""

    # Class-level character mapping
    CHAR_TO_IDX = {chr(97 + i): i for i in range(26)}  # a-z to 0-25
    IDX_TO_CHAR = {i: chr(97 + i) for i in range(26)}  # 0-25 to a-z

    def __init__(
        self,
        data_dir: str | Path,
        image_size: Tuple[int, int] = (28, 28),
        transform: Optional[transforms.Compose] = None,
        is_training: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing character images (e.g., data/characters/regular/train)
            image_size: Target size for images (height, width)
            transform: Optional additional transformations
            is_training: Whether this is a training dataset
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.image_size = image_size
        self.is_training = is_training

        # Get all character directories (a-z)
        self.char_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        if not self.char_dirs:
            raise ValueError(f"No character directories found in {self.data_dir}")

        # Get all image paths and labels
        self.images, self.labels = self._load_dataset()

        # Set up transformations
        self.transform = transform if transform is not None else self._get_default_transforms()

        # Log dataset statistics
        logger.info(f"Loaded {len(self.images)} images from {self.data_dir}")
        char_counts = {char: sum(1 for l in self.labels if l == idx)
                      for char, idx in self.CHAR_TO_IDX.items()}
        logger.info("Character distribution:")
        for char, count in char_counts.items():
            logger.info(f"- {char}: {count} images")

    def _load_dataset(self) -> Tuple[List[Path], List[int]]:
        """Load all image paths and their corresponding labels."""
        images, labels = [], []

        for char_dir in self.char_dirs:
            char = char_dir.name.lower()
            if char not in self.CHAR_TO_IDX:
                logger.warning(f"Skipping unknown character directory: {char}")
                continue

            label = self.CHAR_TO_IDX[char]

            # Get all PNG images in this directory
            char_images = list(char_dir.glob("*.png"))
            images.extend(char_images)
            labels.extend([label] * len(char_images))

        return images, labels

    def _get_default_transforms(self) -> transforms.Compose:
        """Get default transformation pipeline."""
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]

        if self.is_training:
            transform_list.insert(1, transforms.RandomRotation(10))
            transform_list.insert(2, transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ))

        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample."""
        image_path = self.images[idx]
        label = self.labels[idx]

        # Load and convert image
        try:
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a blank image and the label if there's an error
            return torch.zeros((1, *self.image_size)), label

    @classmethod
    def get_char_mapping(cls) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Get the character to index and index to character mappings.

        Returns:
            Tuple of (char_to_idx, idx_to_char) dictionaries
        """
        return cls.CHAR_TO_IDX, cls.IDX_TO_CHAR


def get_data_loaders(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (28, 28),
    num_workers: int = 4,
    augment: bool = True
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training and validation.

    Args:
        data_dir: Path to the data/characters directory (e.g., data/characters)
        batch_size: Batch size for training
        image_size: Target size for images
        num_workers: Number of worker processes for data loading
        augment: Whether to use data augmentation

    Returns:
        Dictionary containing train and validation data loaders
    """
    data_dir = Path(data_dir)

    # Validate directory structure
    validate_data_directory(data_dir)

    # Training datasets (both regular and obfuscated)
    train_regular = CharacterDataset(
        data_dir / "regular" / "train",
        image_size=image_size,
        is_training=True
    )
    train_obfuscated = CharacterDataset(
        data_dir / "obfuscated" / "train",
        image_size=image_size,
        is_training=True
    )

    # Test datasets (both regular and obfuscated)
    test_regular = CharacterDataset(
        data_dir / "regular" / "test",
        image_size=image_size,
        is_training=False
    )
    test_obfuscated = CharacterDataset(
        data_dir / "obfuscated" / "test",
        image_size=image_size,
        is_training=False
    )

    # Combine datasets
    train_dataset = ConcatDataset([train_regular, train_obfuscated])
    test_dataset = ConcatDataset([test_regular, test_obfuscated])

    # Log dataset statistics
    logger.info(f"Combined dataset statistics:")
    logger.info(f"- Training set: {len(train_dataset)} images")
    logger.info(f"- Test set: {len(test_dataset)} images")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        "train": train_loader,
        "test": test_loader
    }


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

    def _parse_image(self, filename: str, split: str):
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

    def _augment(self, obfuscated_img, standard_img):
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
