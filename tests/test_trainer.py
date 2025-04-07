

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import shutil
import yaml
import tempfile
from PIL import Image

from src.train.trainer import ModelTrainer
from src.models.letter_classifier import LetterClassifierCNN
from src.data.data_loader import get_data_loaders
from .test_utils import load_test_config, setup_test_environment, get_test_model, create_test_image

class TestTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment and load configuration."""
        cls.config = load_test_config()
        setup_test_environment(cls.config)

        # Create temporary directory for test data
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.output_dir = cls.test_dir / "outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

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
        self.model = get_test_model(self.config)
        self.train_config = self.config['training']
        self.data_config = self.config['data']

        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            data_dir=self.test_dir,
            batch_size=self.train_config['batch_size'],
            image_size=tuple(self.config['model']['image_size']),
            num_workers=self.data_config['num_workers'],
            augment=self.data_config['augment']
        )

        # Create optimizer and criterion
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.train_config['learning_rate'],
            weight_decay=self.train_config['weight_decay']
        )
        self.criterion = nn.CrossEntropyLoss()

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = ModelTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            output_dir=self.output_dir
        )

        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.model, self.model)
        self.assertEqual(trainer.train_loader, self.train_loader)
        self.assertEqual(trainer.val_loader, self.val_loader)

    def test_train_epoch(self):
        """Test training for one epoch."""
        trainer = ModelTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            output_dir=self.output_dir
        )

        # Train for one epoch
        train_loss, train_acc = trainer.train_epoch()

        # Check if loss and accuracy are valid
        self.assertIsInstance(train_loss, float)
        self.assertIsInstance(train_acc, float)
        self.assertGreaterEqual(train_loss, 0)
        self.assertGreaterEqual(train_acc, 0)
        self.assertLessEqual(train_acc, 100)

    def test_validate(self):
        """Test validation process."""
        trainer = ModelTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            output_dir=self.output_dir
        )

        # Run validation
        val_loss, val_acc = trainer.validate()

        # Check if loss and accuracy are valid
        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(val_acc, float)
        self.assertGreaterEqual(val_loss, 0)
        self.assertGreaterEqual(val_acc, 0)
        self.assertLessEqual(val_acc, 100)

    def test_checkpoint_saving(self):
        """Test model checkpoint saving."""
        trainer = ModelTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            output_dir=self.output_dir
        )

        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint(epoch=1, is_best=False)

        # Check if checkpoint file exists
        self.assertTrue(Path(checkpoint_path).exists())

        # Load checkpoint
        loaded_model = LetterClassifierCNN(
            input_channels=self.config['model']['input_channels'],
            num_classes=self.config['model']['num_classes']
        )
        trainer.load_checkpoint(loaded_model, checkpoint_path)

        # Check if model state was loaded correctly
        self.assertTrue(all(torch.equal(p1, p2) for p1, p2 in zip(
            self.model.parameters(), loaded_model.parameters()
        )))

    def test_early_stopping(self):
        """Test early stopping functionality."""
        trainer = ModelTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            output_dir=self.output_dir,
            early_stopping_patience=self.train_config['early_stopping_patience']
        )

        # Simulate training with no improvement
        for _ in range(self.train_config['early_stopping_patience'] + 1):
            trainer.validate()
            trainer.early_stopping(1.0)  # Constant high loss

        self.assertTrue(trainer.early_stopping.should_stop)

    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling."""
        trainer = ModelTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            output_dir=self.output_dir,
            scheduler_config=self.train_config['scheduler']
        )

        # Initial learning rate
        initial_lr = self.optimizer.param_groups[0]['lr']

        # Simulate validation with no improvement
        for _ in range(self.train_config['scheduler']['patience'] + 1):
            trainer.validate()
            trainer.scheduler.step(1.0)  # High loss

        # Check if learning rate was reduced
        self.assertLess(
            self.optimizer.param_groups[0]['lr'],
            initial_lr
        )

if __name__ == '__main__':
    unittest.main()