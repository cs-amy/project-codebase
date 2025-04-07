

import argparse
import json
import logging
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.letter_classifier import LetterClassifierCNN
from src.utils.config import load_config, get_model_config

def setup_logging(output_dir: Path):
    """Set up logging configuration."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "inference.log"),
            logging.StreamHandler()
        ]
    )

def preprocess_image(image_path: Path, image_size: tuple = (28, 28)):
    """Preprocess a single image for inference."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def plot_prediction(image, pred, true_label=None, output_path: Path = None):
    """Plot the input image and prediction."""
    plt.figure(figsize=(4, 4))
    plt.imshow(image.squeeze(), cmap='gray')
    title = f"Predicted: {chr(pred + ord('a'))}"
    if true_label is not None:
        title += f"\nTrue: {chr(true_label + ord('a'))}"
    plt.title(title)
    plt.axis('off')
    if output_path:
        plt.savefig(output_path)
    plt.close()

def process_single_image(model, image_path: Path, device, output_dir: Path):
    """Process a single image and save the prediction."""
    # Preprocess image
    image = preprocess_image(image_path).to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        pred = pred.item()

    # Save prediction
    result = {
        "image_path": str(image_path),
        "prediction": chr(pred + ord('a')),
        "confidence": torch.softmax(output, 1).max().item()
    }

    # Plot and save visualization
    plot_path = output_dir / "visualizations" / f"{image_path.stem}_pred.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_prediction(image.cpu().squeeze(), pred, output_path=plot_path)

    return result

def process_directory(model, input_dir: Path, device, output_dir: Path):
    """Process all images in a directory."""
    results = []
    image_files = list(input_dir.glob("*.png"))

    for image_path in tqdm(image_files, desc="Processing images"):
        result = process_single_image(model, image_path, device, output_dir)
        results.append(result)

    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run inference on new images")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the trained model checkpoint")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                      help="Path to the configuration file")
    parser.add_argument("--input", type=str, required=True,
                      help="Path to input image or directory of images")
    parser.add_argument("--output_dir", type=str, default="outputs/inference",
                      help="Directory to save inference results")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    args = parser.parse_args()

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load configuration
    config = load_config(args.config)
    model_config = get_model_config(config)

    # Load model
    model = LetterClassifierCNN(
        input_channels=1,
        num_classes=26,
        dropout_rates=model_config.get("dropout_rates", [0.25, 0.25, 0.5, 0.5, 0.5]),
        use_batch_norm=model_config.get("use_batch_norm", True)
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"Model loaded from {args.model_path}")

    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        results = [process_single_image(model, input_path, device, output_dir)]
    else:
        results = process_directory(model, input_path, device, output_dir)

    # Save results
    with open(output_dir / "inference_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Log summary
    logging.info(f"Processed {len(results)} images")
    logging.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()