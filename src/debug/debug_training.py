"""
Debug script for identifying issues with the training pipeline.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Debug training pipeline")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                      help="Path to the configuration file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()
    
    # Print Python and package versions
    logger.info(f"Python version: {sys.version}")
    
    try:
        import numpy as np
        logger.info(f"NumPy version: {np.__version__}")
    except ImportError:
        logger.error("NumPy not installed")
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.error("PyTorch not installed")
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return
    
    logger.info(f"Config file found: {config_path}")
    
    # Try to load config
    try:
        from src.utils.config import load_config
        config = load_config(config_path)
        logger.info("Config loaded successfully")
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return
    
    # Check data directories
    data_config = config.get("data", {})
    data_dirs = []
    
    if "regular" in data_config:
        data_dirs.extend([
            data_config["regular"].get("train_dir"),
            data_config["regular"].get("test_dir")
        ])
    
    if "obfuscated" in data_config:
        data_dirs.extend([
            data_config["obfuscated"].get("train_dir"),
            data_config["obfuscated"].get("test_dir")
        ])
    
    for dir_path in data_dirs:
        if dir_path:
            full_path = Path(dir_path)
            if full_path.exists():
                logger.info(f"Data directory found: {full_path}")
                # Count files
                num_files = len(list(full_path.glob("**/*.png")))
                logger.info(f"  - Contains {num_files} PNG files")
            else:
                logger.warning(f"Data directory not found: {full_path}")
    
    # Try to import and create a small dataset
    try:
        logger.info("Attempting to create a small dataset...")
        from src.data.data_loader import CharacterDataset
        
        # Try to create a dataset with a small sample
        for dir_path in data_dirs:
            if dir_path and Path(dir_path).exists():
                try:
                    logger.info(f"Creating dataset from {dir_path}")
                    dataset = CharacterDataset(
                        data_dir=dir_path,
                        image_size=(28, 28),
                        is_training=True
                    )
                    logger.info(f"Dataset created successfully with {len(dataset)} samples")
                    
                    # Try to get a single sample
                    if len(dataset) > 0:
                        logger.info("Attempting to get a single sample...")
                        sample, label = dataset[0]
                        logger.info(f"Sample shape: {sample.shape}, Label: {label}")
                        break
                except Exception as e:
                    logger.error(f"Error creating dataset from {dir_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Error importing or using CharacterDataset: {str(e)}")
    
    logger.info("Debug completed")

if __name__ == "__main__":
    main()
