import unittest
import torch
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.train.train import train, setup_device, get_optimal_batch_size, resume_training
from src.train.trainer import ModelTrainer
from src.models.letter_classifier import LetterClassifierCNN

class TestTrainingModule(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Sample configuration
        self.config = {
            "model": {
                "architecture": "LetterClassifierCNN",
                "input_shape": [28, 28, 1],
                "num_classes": 26,
                "dropout_rate": 0.5
            },
            "training": {
                "epochs": 2,
                "batch_size": 32,
                "learning_rate": 0.01,
                "weight_decay": 0.0001,
                "optimizer": "adam",
                "lr_scheduler": {
                    "use": True,
                    "type": "reduce_on_plateau",
                    "patience": 5,
                    "factor": 0.5,
                    "min_lr": 0.00001
                },
                "early_stopping": {
                    "use": True,
                    "patience": 15,
                    "min_delta": 0.001
                }
            },
            "data": {
                "regular": {
                    "train_dir": "data/characters/regular/train",
                    "test_dir": "data/characters/regular/test"
                },
                "obfuscated": {
                    "train_dir": "data/characters/obfuscated/train",
                    "test_dir": "data/characters/obfuscated/test"
                },
                "image_size": [28, 28],
                "validation_split": 0.2,
                "shuffle": True,
                "augmentation": {
                    "use": True,
                    "rotation_range": 10,
                    "zoom_range": 0.1,
                    "width_shift_range": 0.1,
                    "height_shift_range": 0.1,
                    "brightness_range": [0.8, 1.2],
                    "random_noise": 0.01
                }
            }
        }

    def test_setup_device(self):
        """Test device setup function."""
        device = setup_device()
        self.assertIsInstance(device, torch.device)

    def test_get_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        # Test with default memory estimation
        batch_size = get_optimal_batch_size((28, 28))
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)
        
        # Test with specific memory
        batch_size = get_optimal_batch_size((28, 28), available_memory_gb=8)
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)

    @patch('src.train.train.ModelTrainer')
    def test_resume_training(self, mock_trainer):
        """Test training resumption functionality."""
        # Create a mock checkpoint file
        checkpoint_path = Path(self.test_dir) / "checkpoint.pth"
        checkpoint_path.touch()
        
        # Mock the trainer instance
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        
        # Test resuming training
        result = resume_training(mock_trainer_instance, checkpoint_path)
        self.assertTrue(result)
        mock_trainer_instance.load_checkpoint.assert_called_once_with(checkpoint_path)

    @patch('src.train.train.get_data_loaders')
    @patch('src.train.train.get_model')
    def test_train_function(self, mock_get_model, mock_get_data_loaders):
        """Test the main training function."""
        # Mock the data loaders
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_test_loader = MagicMock()
        mock_get_data_loaders.return_value = {
            "train": mock_train_loader,
            "val": mock_val_loader,
            "test": mock_test_loader
        }
        
        # Mock the model
        mock_model = MagicMock(spec=LetterClassifierCNN)
        mock_get_model.return_value = mock_model
        
        # Run the training function
        train(self.config)
        
        # Verify that the model and data loaders were created
        mock_get_model.assert_called_once()
        mock_get_data_loaders.assert_called_once()

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for ModelTrainer."""
        self.model = MagicMock(spec=LetterClassifierCNN)
        self.train_loader = MagicMock()
        self.test_loader = MagicMock()
        self.config = {
            "training": {
                "learning_rate": 0.01,
                "lr_scheduler": {
                    "factor": 0.5,
                    "patience": 5,
                    "min_lr": 0.00001
                },
                "early_stopping": {
                    "patience": 15
                }
            }
        }
        self.output_dir = Path(tempfile.mkdtemp())
        
        # Create trainer instance
        self.trainer = ModelTrainer(
            model=self.model,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            config=self.config,
            output_dir=self.output_dir
        )

    def test_train_epoch(self):
        """Test training for one epoch."""
        # Mock the data loader to return some dummy data
        self.train_loader.__iter__.return_value = [
            (torch.randn(2, 1, 28, 28), torch.tensor([0, 1])),
            (torch.randn(2, 1, 28, 28), torch.tensor([2, 3]))
        ]
        
        # Mock the model's forward pass
        self.model.return_value = torch.randn(2, 26)
        
        # Run one training epoch
        loss, acc = self.trainer.train_epoch()
        
        # Verify the results
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 100)

    def test_validate(self):
        """Test validation function."""
        # Mock the test loader to return some dummy data
        self.test_loader.__iter__.return_value = [
            (torch.randn(2, 1, 28, 28), torch.tensor([0, 1])),
            (torch.randn(2, 1, 28, 28), torch.tensor([2, 3]))
        ]
        
        # Mock the model's forward pass
        self.model.return_value = torch.randn(2, 26)
        
        # Run validation
        loss, acc = self.trainer.validate()
        
        # Verify the results
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 100)

    def test_save_checkpoint(self):
        """Test checkpoint saving functionality."""
        # Create a checkpoint
        self.trainer.save_checkpoint(epoch=1, is_best=True)
        
        # Verify that checkpoint files were created
        self.assertTrue((self.output_dir / "checkpoint_epoch_1.pth").exists())
        self.assertTrue((self.output_dir / "best_model.pth").exists())

    def test_load_checkpoint(self):
        """Test checkpoint loading functionality."""
        # Create a dummy checkpoint
        checkpoint = {
            'epoch': 1,
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'scheduler_state_dict': {},
            'best_val_loss': 0.5,
            'history': {'train_loss': [], 'val_loss': []}
        }
        
        checkpoint_path = self.output_dir / "test_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Load the checkpoint
        epoch = self.trainer.load_checkpoint(checkpoint_path)
        
        # Verify the loaded epoch
        self.assertEqual(epoch, 1)

if __name__ == '__main__':
    unittest.main()
