"""
Script for training the deobfuscation model.
"""

import os
import sys
import argparse
import torch
import logging
from pathlib import Path
import yaml
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import load_config, get_model_config, get_training_config, get_data_config
from data.data_loader import get_data_loaders
from models.deobfuscator_cnn import get_model
from trainer import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train deobfuscation model")
    
    parser.add_argument("--config", type=str, default="configs/model_config.yaml",
                       help="Path to the config file")
    parser.add_argument("--data_dir", type=str, default="data/characters",
                       help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Path to the output directory")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model architecture name (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate (overrides config)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU for training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()


def main():
    """Main function for training the model."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        print(f"Config file not found: {config_path}")
        return
    
    config = load_config(config_path)
    model_config = get_model_config(config)
    training_config = get_training_config(config)
    data_config = get_data_config(config)
    
    # Override config with command line arguments
    if args.model_name:
        model_config["architecture"] = args.model_name
    if args.batch_size:
        training_config["batch_size"] = args.batch_size
    if args.epochs:
        training_config["epochs"] = args.epochs
    if args.learning_rate:
        training_config["learning_rate"] = args.learning_rate
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config with any command line overrides
    config["model"] = model_config
    config["training"] = training_config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create data loaders
    logger.info(f"Loading data from {args.data_dir}")
    print(f"Loading data from {args.data_dir}")
    data_loaders = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=training_config["batch_size"],
        image_size=model_config["input_shape"][:2],
        num_workers=4,
        augment=data_config.get("augmentation", {}).get("use", True)
    )
    
    # Create model
    logger.info(f"Creating model: {model_config['architecture']}")
    print(f"Creating model: {model_config['architecture']}")
    model = get_model(model_config["architecture"], model_config)
    
    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    print(f"Using device: {device}")
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=data_loaders["train"],
        val_loader=data_loaders["val"],
        config=training_config,
        output_dir=output_dir,
        device=device
    )
    
    # Resume training if requested
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    logger.info("Starting training...")
    print("Starting training...")
    try:
        history = trainer.train()
        
        # Save training history
        with open(output_dir / "history.json", "w") as f:
            history_serializable = {k: [float(x) for x in v] for k, v in history.items()}
            json.dump(history_serializable, f, indent=2)
        
        logger.info(f"Training completed, results saved to {output_dir}")
        print(f"Training completed, results saved to {output_dir}")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("Training interrupted by user")
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        print(f"Error during training: {e}")
    # Save final model regardless of how training ended
    trainer.save_model("final")
    logger.info(f"Final model saved to {output_dir}/model_final.pth")
    print(f"Final model saved to {output_dir}/model_final.pth")

if __name__ == "__main__":
    main()
