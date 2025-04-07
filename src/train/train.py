import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.data_loader import CharacterDataset, get_data_loaders
from src.models.letter_classifier import LetterClassifierCNN
from src.train.trainer import ModelTrainer
from src.utils.config import load_config, get_model_config, get_training_config, get_data_config

def setup_logging(output_dir: Path):
    """Set up logging configuration."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler()
        ]
    )

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train the character deobfuscation model")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                      help="Path to the configuration file")
    parser.add_argument("--output_dir", type=str, default="outputs/letter_classifier",
                      help="Directory to save training outputs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    model_config = get_model_config(config)
    training_config = get_training_config(config)
    data_config = get_data_config(config)

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Get data loaders
    train_loader, val_loader = get_data_loaders(data_config)
    logging.info(f"Training set size: {len(train_loader.dataset)}")
    logging.info(f"Validation set size: {len(val_loader.dataset)}")

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
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        **training_config
    )

    # Train the model
    try:
        trainer.train()
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()