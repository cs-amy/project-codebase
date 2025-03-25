"""
Script for evaluating the deobfuscation model.
"""

import os
import sys
import argparse
import torch
import logging
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config, get_model_config, get_inference_config
from src.data.data_loader import get_data_loaders
from src.models.deobfuscator_cnn import get_model
from src.evaluate.evaluator import Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate deobfuscation model")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to the model config file")
    parser.add_argument("--data_dir", type=str, default="data/characters",
                       help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Path to the output directory")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Threshold for binarizing predictions (overrides config)")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of sample visualizations to save")
    parser.add_argument("--pipeline_comparison", action="store_true",
                       help="Perform OCR pipeline comparison")
    parser.add_argument("--confusion_matrix", action="store_true",
                       help="Generate confusion matrix for OCR results")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU for evaluation")
    
    return parser.parse_args()


def main():
    """Main function for evaluating the model."""
    args = parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return
    
    config = load_config(config_path)
    model_config = get_model_config(config)
    inference_config = get_inference_config(config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create data loaders
    logger.info(f"Loading test data from {args.data_dir}")
    data_loaders = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=model_config["input_shape"][:2],
        num_workers=4,
        augment=False  # No augmentation for evaluation
    )
    
    # Create model
    logger.info(f"Creating model: {model_config['architecture']}")
    model = get_model(model_config["architecture"], model_config)
    
    # Load model weights
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        return
    
    logger.info(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set threshold
    threshold = args.threshold
    if threshold is None:
        threshold = inference_config.get("confidence_threshold", 0.5)
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=data_loaders["test"],
        device=device,
        output_dir=output_dir,
        threshold=threshold
    )
    
    # Evaluate model
    logger.info("Starting evaluation...")
    results = evaluator.evaluate(
        save_results=True,
        save_samples=True,
        num_samples=args.num_samples,
        ocr_evaluation=True
    )
    
    # Perform additional evaluations if requested
    if args.pipeline_comparison:
        logger.info("Performing OCR pipeline comparison...")
        # This would require additional implementation for real-world test cases
        # For now, we'll just log a message
        logger.info("Pipeline comparison is not implemented in this script")
        
    if args.confusion_matrix:
        logger.info("Generating confusion matrix...")
        evaluator.create_confusion_matrix(save_results=True)
    
    logger.info(f"Evaluation completed, results saved to {output_dir}")


if __name__ == "__main__":
    main() 