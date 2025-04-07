

import unittest
import torch
import numpy as np
from pathlib import Path
import shutil
import tempfile
from sklearn.metrics import confusion_matrix, classification_report

from src.evaluate.evaluate import ModelEvaluator
from src.models.letter_classifier import LetterClassifierCNN

class TestEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directory for test data
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.output_dir = cls.test_dir / "outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        """Set up test case specific variables."""
        self.num_classes = 3
        self.batch_size = 4
        self.model = LetterClassifierCNN(
            input_channels=1,
            num_classes=self.num_classes
        )
        self.evaluator = ModelEvaluator(
            model=self.model,
            output_dir=self.output_dir
        )

    def test_confusion_matrix(self):
        """Test confusion matrix generation."""
        # Create sample predictions and targets
        predictions = torch.tensor([0, 1, 2, 1])
        targets = torch.tensor([0, 1, 2, 0])

        # Generate confusion matrix
        cm = self.evaluator.create_confusion_matrix(predictions, targets)

        # Check matrix properties
        self.assertEqual(cm.shape, (self.num_classes, self.num_classes))
        self.assertTrue(np.all(cm >= 0))
        self.assertEqual(np.sum(cm), len(predictions))

    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        # Create sample predictions and targets
        predictions = torch.tensor([0, 1, 2, 1])
        targets = torch.tensor([0, 1, 2, 0])

        # Calculate accuracy
        accuracy = self.evaluator.calculate_accuracy(predictions, targets)

        # Check accuracy properties
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

    def test_per_class_accuracy(self):
        """Test per-class accuracy calculation."""
        # Create sample predictions and targets
        predictions = torch.tensor([0, 1, 2, 1])
        targets = torch.tensor([0, 1, 2, 0])

        # Calculate per-class accuracy
        per_class_acc = self.evaluator.calculate_per_class_accuracy(predictions, targets)

        # Check per-class accuracy properties
        self.assertEqual(len(per_class_acc), self.num_classes)
        self.assertTrue(all(0 <= acc <= 1 for acc in per_class_acc))

    def test_loss_calculation(self):
        """Test loss calculation."""
        # Create sample predictions and targets
        predictions = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))

        # Calculate loss
        loss = self.evaluator.calculate_loss(predictions, targets)

        # Check loss properties
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

    def test_evaluation_results_saving(self):
        """Test saving evaluation results."""
        # Create sample evaluation results
        results = {
            'accuracy': 0.85,
            'loss': 0.5,
            'per_class_accuracy': [0.8, 0.9, 0.85],
            'confusion_matrix': np.random.rand(3, 3)
        }

        # Save results
        self.evaluator.save_results(results)

        # Check if files were created
        self.assertTrue((self.output_dir / "evaluation_results.json").exists())
        self.assertTrue((self.output_dir / "confusion_matrix.png").exists())

    def test_classification_report(self):
        """Test classification report generation."""
        # Create sample predictions and targets
        predictions = torch.tensor([0, 1, 2, 1])
        targets = torch.tensor([0, 1, 2, 0])

        # Generate classification report
        report = self.evaluator.generate_classification_report(predictions, targets)

        # Check report properties
        self.assertIsInstance(report, str)
        self.assertIn('precision', report)
        self.assertIn('recall', report)
        self.assertIn('f1-score', report)

    def test_visualization(self):
        """Test visualization generation."""
        # Create sample data for visualization
        train_losses = [0.5, 0.4, 0.3]
        val_losses = [0.6, 0.5, 0.4]
        train_accs = [0.7, 0.8, 0.9]
        val_accs = [0.6, 0.7, 0.8]

        # Generate visualizations
        self.evaluator.plot_training_history(
            train_losses, val_losses, train_accs, val_accs
        )

        # Check if plot was saved
        self.assertTrue((self.output_dir / "training_history.png").exists())

if __name__ == '__main__':
    unittest.main()