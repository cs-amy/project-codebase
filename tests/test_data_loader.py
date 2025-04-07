import unittest
import torch
from pathlib import Path
import shutil
from PIL import Image

from src.data.data_loader import CharacterDataset, get_data_loaders
from .test_utils import load_test_config, setup_test_environment, create_test_image

class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment and load configuration."""
        cls.config = load_test_config()
        setup_test_environment(cls.config)
        
        # Create test data directory
        cls.test_dir = Path(cls.config['data']['test_data_dir'])
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample test images
        cls.create_test_dataset()

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if cls.config['environment']['cleanup_after_test']:
            shutil.rmtree(cls.test_dir)

    @classmethod
    def create_test_dataset(cls):
        """Create a small test dataset with equal samples per class."""
        data_config = cls.config['data']
        num_images = data_config['num_test_images']
        test_chars = data_config['test_characters']
        
        # Create directories for each character
        for char in test_chars:
            char_dir = cls.test_dir / char
            char_dir.mkdir(parents=True, exist_ok=True)
            
            # Create sample images
            for i in range(num_images):
                img = create_test_image(cls.config)
                img = (img * 255).byte().squeeze().numpy()
                img = Image.fromarray(img)
                img.save(char_dir / f"{i:03d}.png")

    def setUp(self):
        """Set up test case specific variables."""
        self.data_config = self.config['data']
        self.image_size = tuple(self.config['model']['image_size'])
        self.batch_size = self.data_config['batch_size']

    def test_dataset_initialization(self):
        """Test dataset initialization with valid paths."""
        dataset = CharacterDataset(
            root_dir=self.test_dir,
            transform=None,
            image_size=self.image_size
        )
        self.assertEqual(len(dataset), len(self.data_config['test_characters']) * self.data_config['num_test_images'])
        self.assertEqual(dataset.num_classes, len(self.data_config['test_characters']))

    def test_dataset_invalid_path(self):
        """Test dataset initialization with invalid path."""
        with self.assertRaises(ValueError):
            CharacterDataset(
                root_dir="nonexistent/path",
                transform=None,
                image_size=self.image_size
            )

    def test_dataset_get_item(self):
        """Test dataset item retrieval."""
        dataset = CharacterDataset(
            root_dir=self.test_dir,
            transform=None,
            image_size=self.image_size
        )
        
        # Get a sample
        image, label = dataset[0]
        
        # Check types and shapes
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, int)
        self.assertEqual(image.shape, (1, *self.image_size))
        self.assertGreaterEqual(label, 0)
        self.assertLess(label, len(self.data_config['test_characters']))

    def test_data_loaders(self):
        """Test data loader creation and functionality."""
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir=self.test_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
            num_workers=self.data_config['num_workers'],
            augment=self.data_config['augment']
        )
        
        # Check if all loaders are created
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        
        # Test batch loading
        batch = next(iter(train_loader))
        self.assertEqual(len(batch), 2)  # images and labels
        self.assertEqual(batch[0].shape[0], self.batch_size)
        self.assertEqual(batch[1].shape[0], self.batch_size)

    def test_data_augmentation(self):
        """Test data augmentation when enabled."""
        dataset = CharacterDataset(
            root_dir=self.test_dir,
            transform=None,
            image_size=self.image_size,
            augment=True
        )
        
        # Get same item twice
        image1, _ = dataset[0]
        image2, _ = dataset[0]
        
        # Images should be different due to augmentation
        self.assertFalse(torch.allclose(image1, image2))

    def test_character_mapping(self):
        """Test character to index mapping consistency."""
        dataset = CharacterDataset(
            root_dir=self.test_dir,
            transform=None,
            image_size=self.image_size
        )
        
        # Check if mapping is consistent
        char_to_idx = dataset.char_to_idx
        self.assertEqual(len(char_to_idx), len(self.data_config['test_characters']))
        
        # Check if all test characters are in mapping
        for char in self.data_config['test_characters']:
            self.assertIn(char, char_to_idx)

if __name__ == '__main__':
    unittest.main() 