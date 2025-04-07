

import unittest
import torch
import numpy as np
from pathlib import Path
import shutil
import tempfile
from PIL import Image

from src.inference.inference import ModelInference
from src.models.letter_classifier import LetterClassifierCNN

class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directory for test data
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.output_dir = cls.test_dir / "outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

        # Create sample test images
        cls.test_images_dir = cls.test_dir / "test_images"
        cls.test_images_dir.mkdir(parents=True, exist_ok=True)

        # Create a sample image
        img = Image.new('L', (28, 28), color=0)
        img.save(cls.test_images_dir / "test_char.png")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        """Set up test case specific variables."""
        self.num_classes = 26
        self.model = LetterClassifierCNN(
            input_channels=1,
            num_classes=self.num_classes
        )
        self.inference = ModelInference(
            model=self.model,
            output_dir=self.output_dir
        )

    def test_image_preprocessing(self):
        """Test image preprocessing."""
        # Load test image
        image_path = self.test_images_dir / "test_char.png"
        image = self.inference.preprocess_image(image_path)

        # Check image properties
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (1, 1, 28, 28))
        self.assertTrue(torch.all(image >= 0) and torch.all(image <= 1))

    def test_single_prediction(self):
        """Test single image prediction."""
        # Load and preprocess test image
        image_path = self.test_images_dir / "test_char.png"
        image = self.inference.preprocess_image(image_path)

        # Get prediction
        prediction = self.inference.predict_single(image)

        # Check prediction properties
        self.assertIsInstance(prediction, dict)
        self.assertIn('class', prediction)
        self.assertIn('confidence', prediction)
        self.assertIn('probabilities', prediction)
        self.assertEqual(len(prediction['probabilities']), self.num_classes)

    def test_batch_prediction(self):
        """Test batch prediction."""
        # Create batch of test images
        batch_size = 4
        images = torch.randn(batch_size, 1, 28, 28)

        # Get predictions
        predictions = self.inference.predict_batch(images)

        # Check predictions properties
        self.assertEqual(len(predictions), batch_size)
        for pred in predictions:
            self.assertIsInstance(pred, dict)
            self.assertIn('class', pred)
            self.assertIn('confidence', pred)
            self.assertIn('probabilities', pred)

    def test_inference_pipeline(self):
        """Test complete inference pipeline."""
        # Run inference on test image
        image_path = self.test_images_dir / "test_char.png"
        result = self.inference.run_inference(image_path)

        # Check result properties
        self.assertIsInstance(result, dict)
        self.assertIn('predictions', result)
        self.assertIn('visualization_path', result)

        # Check if visualization was saved
        self.assertTrue(Path(result['visualization_path']).exists())

    def test_inference_with_threshold(self):
        """Test inference with confidence threshold."""
        # Set confidence threshold
        threshold = 0.8

        # Run inference with threshold
        image_path = self.test_images_dir / "test_char.png"
        result = self.inference.run_inference(image_path, confidence_threshold=threshold)

        # Check that predictions respect threshold
        for pred in result['predictions']:
            self.assertGreaterEqual(pred['confidence'], threshold)

    def test_inference_error_handling(self):
        """Test inference error handling."""
        # Test with non-existent image
        with self.assertRaises(FileNotFoundError):
            self.inference.run_inference("nonexistent.png")

        # Test with invalid image
        invalid_image = self.test_images_dir / "invalid.png"
        invalid_image.touch()
        with self.assertRaises(Exception):
            self.inference.run_inference(invalid_image)

    def test_visualization_generation(self):
        """Test visualization generation."""
        # Run inference with visualization
        image_path = self.test_images_dir / "test_char.png"
        result = self.inference.run_inference(image_path, save_visualization=True)

        # Check visualization file
        self.assertTrue(Path(result['visualization_path']).exists())

        # Check visualization content
        viz_image = Image.open(result['visualization_path'])
        self.assertEqual(viz_image.size, (28, 28))

if __name__ == '__main__':
    unittest.main()