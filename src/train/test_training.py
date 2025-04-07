"""
Script for running a test training to verify the training pipeline.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import Subset, DataLoader

from src.data.data_loader import CharacterDataset, get_data_loaders
from src.models.letter_classifier import LetterClassifierCNN
from src.train.trainer import ModelTrainer
from src.utils.config import load_config, get_model_config, get_training_config, get_data_config

def setup_test_environment(config: dict, output_dir: Path):
    """Set up a test environment with a small dataset."""
    # Create test output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_output_dir = output_dir / f"test_runs/test_{timestamp}"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_dir = test_output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "test_training.log"),
            logging.StreamHandler()
        ]
    )

    # Modify config for testing
    config["training"]["epochs"] = 5
    config["training"]["batch_size"] = 32
    config["training"]["early_stopping"]["patience"] = 3
    config["data"]["train_split"] = 0.8

    return test_output_dir, config

def create_test_dataset(dataset: CharacterDataset, samples_per_class: int = 5):
    """Create a small test dataset with equal samples per class."""
    indices = []
    for class_idx in range(26):  # 26 letters
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]
        indices.extend(class_indices[:samples_per_class])
    return Subset(dataset, indices)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run test training with a small dataset")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                      help="Path to the configuration file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                      help="Base directory for test outputs")
    parser.add_argument("--samples_per_class", type=int, default=5,
                      help="Number of samples per class for training")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    model_config = get_model_config(config)
    training_config = get_training_config(config)
    data_config = get_data_config(config)

    # Set up test environment
    test_output_dir, config = setup_test_environment(config, Path(args.output_dir))
    logging.info(f"Test output directory: {test_output_dir}")

    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Get data loaders
    train_loader, val_loader = get_data_loaders(data_config)

    # Create small test datasets
    test_train_dataset = create_test_dataset(train_loader.dataset, args.samples_per_class)
    test_val_dataset = create_test_dataset(val_loader.dataset, args.samples_per_class // 5)

    test_train_loader = DataLoader(
        test_train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0 # temporarily set to 0
    )

    test_val_loader = DataLoader(
        test_val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0 # temporarily set to 0
    )

    logging.info(f"Test training set size: {len(test_train_loader.dataset)}")
    logging.info(f"Test validation set size: {len(test_val_loader.dataset)}")

    # Initialize model
    model = LetterClassifierCNN(
        input_channels=1,
        num_classes=26,
        dropout_rates=model_config.get("dropout_rates", [0.25, 0.25, 0.5, 0.5, 0.5]),
        use_batch_norm=model_config.get("use_batch_norm", True)
    ).to(device)
    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=test_train_loader,
        val_loader=test_val_loader,
        device=device,
        output_dir=test_output_dir,
        **training_config
    )

    # Train the model
    try:
        trainer.train()
        logging.info("Test training completed successfully")
    except Exception as e:
        logging.error(f"Test training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()