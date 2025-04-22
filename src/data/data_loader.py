"""
Data loader module for loading and processing image pairs for model training.
"""

from pathlib import Path
from typing import Dict, Tuple
from rich.console import Console
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch

from src.models.character_dataset import CharacterDataset

# Initialize rich console
console = Console()

# Configure logging
# Routine for validating directory structure
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
    console.print("[green]Data directory structure validation successful![/green]")
    console.print(f"Root: {data_dir}")
    for dir_path in required_dirs:
        full_path = data_dir / dir_path
        num_files = len(list(full_path.glob("**/*.png")))
        console.print(f"- {dir_path}: {num_files} PNG files")

def show_images(dataset, title):
    """
    Display a grid of images from the dataset.
    Args:
        dataset: Dataset object containing images and labels
        title: Title for the plot
    """
    if len(dataset) < 3:
        raise ValueError("Dataset must contain at least 3 images to display.")
    
    for i in range(3):
        raw_img, label = dataset[i]
        img = mpimg.imread(raw_img)
        imgplot = plt.imshow(img)
        plt.show()

def get_data_loaders(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (28, 28),
    num_workers: int = 4,
    augment: bool = True,
    limit_train_samples: int = None
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        data_dir: Path to the data/characters directory (e.g., data/characters)
        batch_size: Batch size for training
        image_size: Target size for images
        num_workers: Number of worker processes for data loading
        augment: Whether to use data augmentation (applied in training mode)
        limit_train_samples: Optional limit for the number of training samples

    Returns:
        Dictionary containing train, val, and test data loaders.
    """
    data_dir = Path(data_dir)

    # Validate directory structure
    validate_data_directory(data_dir)

    # Load training datasets for both regular and obfuscated characters
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

    # Load test datasets for both regular and obfuscated characters
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

    # Combine datasets for training and testing
    train_dataset = ConcatDataset([train_regular, train_obfuscated])
    test_dataset = ConcatDataset([test_regular, test_obfuscated])

    # Limit the number of samples in the training dataset if requested
    if limit_train_samples is not None and limit_train_samples < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:limit_train_samples]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    # Create train/validation split (80/20) on the (possibly limited) training dataset
    train_length = int(0.8 * len(train_dataset))
    val_length = len(train_dataset) - train_length
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_length, val_length])

    # Show some of the images for each subset
    show_images(train_subset, "Training Subset")
    show_images(val_subset, "Validation Subset")
    show_images(test_dataset, "Test Subset")    

    # Create data loaders for each split
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
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
        "val": val_loader,
        "test": test_loader
    }