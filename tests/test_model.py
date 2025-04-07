

import unittest
import torch
import torch.nn as nn
from src.models.letter_classifier import LetterClassifierCNN
from .test_utils import load_test_config, setup_test_environment, get_test_model

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment and load configuration."""
        cls.config = load_test_config()
        setup_test_environment(cls.config)

    def setUp(self):
        """Set up test case specific variables."""
        self.model = get_test_model(self.config)
        self.model_config = self.config['model']
        self.input_channels = self.model_config['input_channels']
        self.num_classes = self.model_config['num_classes']
        self.image_size = self.model_config['image_size']
        self.batch_size = self.config['inference']['batch_size']

    def test_model_initialization(self):
        """Test model initialization with different configurations."""
        # Test with default configuration
        model = LetterClassifierCNN(
            input_channels=self.input_channels,
            num_classes=self.num_classes
        )
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.num_classes, self.num_classes)

        # Test with custom dropout rates
        custom_dropout = [0.3, 0.3, 0.4, 0.4, 0.4]
        model = LetterClassifierCNN(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            dropout_rates=custom_dropout
        )
        self.assertEqual(len(model.dropout_layers), len(custom_dropout))

    def test_model_forward_pass(self):
        """Test model forward pass with different input sizes."""
        model = LetterClassifierCNN(
            input_channels=self.input_channels,
            num_classes=self.num_classes
        )

        # Test with single image
        x = torch.randn(1, self.input_channels, *self.image_size)
        output = model(x)
        self.assertEqual(output.shape, (1, self.num_classes))

        # Test with batch
        x = torch.randn(self.batch_size, self.input_channels, *self.image_size)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_model_output_range(self):
        """Test model output range and shape."""
        model = LetterClassifierCNN(
            input_channels=self.input_channels,
            num_classes=self.num_classes
        )
        model.eval()  # Set to evaluation mode

        with torch.no_grad():
            x = torch.randn(1, self.input_channels, *self.image_size)
            output = model(x)

            # Check output shape
            self.assertEqual(output.shape, (1, self.num_classes))

            # Check if output contains valid logits
            self.assertTrue(torch.isfinite(output).all())
            self.assertFalse(torch.isnan(output).any())

    def test_model_gradient_flow(self):
        """Test if gradients flow through the model."""
        model = LetterClassifierCNN(
            input_channels=self.input_channels,
            num_classes=self.num_classes
        )
        model.train()  # Set to training mode

        x = torch.randn(1, self.input_channels, *self.image_size, requires_grad=True)
        output = model(x)

        # Check if output requires gradients
        self.assertTrue(output.requires_grad)

        # Test backward pass
        loss = output.mean()
        loss.backward()

        # Check if gradients are computed
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_model_parameters(self):
        """Test model parameter initialization and count."""
        model = LetterClassifierCNN(
            input_channels=self.input_channels,
            num_classes=self.num_classes
        )

        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)

        # Check if all parameters are properly initialized
        for name, param in model.named_parameters():
            self.assertTrue(torch.isfinite(param).all())
            self.assertFalse(torch.isnan(param).any())

    def test_model_dropout(self):
        """Test dropout behavior during training and evaluation."""
        model = LetterClassifierCNN(
            input_channels=self.input_channels,
            num_classes=self.num_classes
        )

        x = torch.randn(1, self.input_channels, *self.image_size)

        # Test in training mode
        model.train()
        output_train = model(x)

        # Test in evaluation mode
        model.eval()
        output_eval = model(x)

        # Outputs should be different due to dropout
        self.assertFalse(torch.allclose(output_train, output_eval))

if __name__ == '__main__':
    unittest.main()