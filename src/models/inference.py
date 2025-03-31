"""
Script for running inference with the trained model on new images.
"""

import os
import sys
import argparse
import torch
import logging
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config, get_model_config, get_inference_config
from src.models.deobfuscator_cnn import get_model
from src.utils.image_processing import resize_image, normalize_image
from src.utils.ocr import recognize_single_character, recognize_word

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to the model config file")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input image or directory of images")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                       help="Path to output directory")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Threshold for binarizing predictions (overrides config)")
    parser.add_argument("--word_mode", action="store_true",
                       help="Process inputs as words instead of single characters")
    parser.add_argument("--perform_ocr", action="store_true",
                       help="Perform OCR on the input and output images")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU for inference")
    
    return parser.parse_args()


def process_image(
    model: torch.nn.Module,
    image_path: str,
    output_dir: Path,
    threshold: float,
    image_size: tuple,
    word_mode: bool,
    perform_ocr: bool,
    device: torch.device
) -> dict:
    """
    Process a single image with the model.
    
    Args:
        model: PyTorch model
        image_path: Path to the image
        output_dir: Output directory
        threshold: Threshold for binarizing predictions
        image_size: Input size for the model as (height, width)
        word_mode: Whether to process as word instead of single character
        perform_ocr: Whether to perform OCR
        device: Device to run inference on
        
    Returns:
        Dictionary of results
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
    
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    
    # Resize to model input size
    image_resized = resize_image(image_gray, image_size)
    
    # Normalize
    image_norm = normalize_image(image_resized)
    
    # Create tensor
    image_tensor = torch.tensor(image_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
    
    # Convert output to numpy
    output_np = output.squeeze().cpu().numpy()
    
    # Thresholding
    _, output_bin = cv2.threshold(
        (output_np * 255).astype(np.uint8),
        int(threshold * 255),
        255,
        cv2.THRESH_BINARY
    )
    
    # Perform OCR if requested
    ocr_results = {}
    if perform_ocr:
        # Convert to uint8 for OCR
        input_img = (image_norm * 255).astype(np.uint8)
        
        if word_mode:
            input_text = recognize_word(input_img)
            output_text = recognize_word(output_bin)
            
            ocr_results = {
                "input_text": input_text,
                "output_text": output_text
            }
        else:
            input_char = recognize_single_character(input_img)
            output_char = recognize_single_character(output_bin)
            
            ocr_results = {
                "input_char": input_char,
                "output_char": output_char
            }
    
    # Save results
    image_name = Path(image_path).stem
    
    # Save output image
    output_path = output_dir / f"{image_name}_deobfuscated.png"
    cv2.imwrite(str(output_path), output_bin)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Input image
    axes[0].imshow(image_resized, cmap='gray')
    axes[0].set_title('Input (Obfuscated)')
    axes[0].axis('off')
    
    # Output image
    axes[1].imshow(output_bin, cmap='gray')
    axes[1].set_title('Output (Deobfuscated)')
    axes[1].axis('off')
    
    # Add OCR results if available
    if perform_ocr:
        if word_mode:
            fig.suptitle(f"Input OCR: '{ocr_results['input_text']}' → Output OCR: '{ocr_results['output_text']}'")
        else:
            fig.suptitle(f"Input OCR: '{ocr_results['input_char']}' → Output OCR: '{ocr_results['output_char']}'")
    
    # Save visualization
    vis_path = output_dir / f"{image_name}_visualization.png"
    plt.tight_layout()
    plt.savefig(vis_path, dpi=150)
    plt.close()
    
    logger.info(f"Processed {image_path} → {output_path}")
    
    return {
        "input_path": image_path,
        "output_path": str(output_path),
        "visualization_path": str(vis_path),
        "ocr_results": ocr_results
    }


def main():
    """Main function for running inference."""
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
    model.to(device)
    model.eval()
    
    # Set threshold
    threshold = args.threshold
    if threshold is None:
        threshold = inference_config.get("confidence_threshold", 0.5)
    
    # Get model input size
    image_size = tuple(model_config["input_shape"][:2])
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single image
        process_image(
            model=model,
            image_path=str(input_path),
            output_dir=output_dir,
            threshold=threshold,
            image_size=image_size,
            word_mode=args.word_mode,
            perform_ocr=args.perform_ocr,
            device=device
        )
    elif input_path.is_dir():
        # Process directory of images
        results = []
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
        for file in input_path.iterdir():
            if file.suffix.lower() in image_extensions:
                result = process_image(
                    model=model,
                    image_path=str(file),
                    output_dir=output_dir,
                    threshold=threshold,
                    image_size=image_size,
                    word_mode=args.word_mode,
                    perform_ocr=args.perform_ocr,
                    device=device
                )
                if result:
                    results.append(result)
        
        logger.info(f"Processed {len(results)} images")
    else:
        logger.error(f"Input path not found: {input_path}")
        return
    
    logger.info(f"Inference completed, results saved to {output_dir}")


if __name__ == "__main__":
    main() 
