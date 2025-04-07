import argparse
import json
import logging
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.data.data_loader import get_data_loaders
from src.models.letter_classifier import LetterClassifierCNN
from src.utils.config import load_config, get_model_config, get_data_config

def setup_logging(output_dir: Path):
    """Set up logging configuration."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "evaluation.log"),
            logging.StreamHandler()
        ]
    )

def plot_confusion_matrix(cm, output_dir: Path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

def evaluate_model(model, data_loader, device):
    """Evaluate model on the given data loader."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), total_loss / len(data_loader)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate the trained model")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the trained model checkpoint")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                      help="Path to the configuration file")
    parser.add_argument("--data_dir", type=str, default="data/characters",
                      help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation",
                      help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for evaluation")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for evaluation")
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
    data_config = get_data_config(config)
    data_config["batch_size"] = args.batch_size

    # Get data loaders
    _, test_loader = get_data_loaders(data_config)
    logging.info(f"Test set size: {len(test_loader.dataset)}")

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

    # Evaluate model
    preds, labels, test_loss = evaluate_model(model, test_loader, device)

    # Calculate metrics
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, output_dict=True)
    accuracy = np.mean(preds == labels)

    # Save results
    results = {
        "test_loss": test_loss,
        "accuracy": accuracy,
        "classification_report": report
    }

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Plot confusion matrix
    plot_confusion_matrix(cm, output_dir)

    # Log results
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info("\nClassification Report:")
    logging.info(classification_report(labels, preds))

if __name__ == "__main__":
    main()