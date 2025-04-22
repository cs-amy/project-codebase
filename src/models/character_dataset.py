from pathlib import Path
from typing import Dict, Tuple, Optional, List
from rich.console import Console
# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image

# Initialize rich console
console = Console()

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
        console.print(f"[green]Loaded {len(self.images)} images from {self.data_dir}[/green]")
        char_counts = {char: sum(1 for l in self.labels if l == idx)
                       for char, idx in self.CHAR_TO_IDX.items()}
        console.print("[green]Character distribution:[/green]")
        for char, count in char_counts.items():
            console.print(f"- {char}: {count} images")

    def _load_dataset(self) -> Tuple[List[Path], List[int]]:
        """Load all image paths and their corresponding labels."""
        images, labels = [], []

        for char_dir in self.char_dirs:
            char = char_dir.name.lower()
            if char not in self.CHAR_TO_IDX:
                console.print(f"[yellow]Skipping unknown character directory: {char}[/yellow]")
                continue

            label = self.CHAR_TO_IDX[char]
            console.print(f"[blue]{char}: Loading images for character[/blue]")

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
            console.print(f"[red]Error loading image {image_path}: {e}[/red]")
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
