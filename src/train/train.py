"""
This module contains the runner function for training the letter classifier model.
"""

import os
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict
from rich.console import Console

from src.data.data_loader import get_data_loaders
from src.models.letter_classifier import get_model
from src.train.trainer import ModelTrainer

# Initialize rich console
console = Console()

def setup_device():
    """
    Set up the training device (GPU/CPU) based on availability.
    Uses MPS for Apple Silicon, CUDA for NVIDIA GPUs, or falls back to CPU.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        console.print("[green]GPU available: Using Metal Performance Shaders (MPS)[/green]")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        console.print("[green]GPU available: Using CUDA[/green]")
    else:
        device = torch.device("cpu")
        console.print("[yellow]No GPU detected: Using CPU[/yellow]")
    return device

def get_optimal_batch_size(image_size, available_memory_gb=None):
    """
    Calculate optimal batch size based on available memory and image size.

    Args:
        image_size (tuple): Image dimensions (height, width)
        available_memory_gb (float): Available GPU memory in GB. If None, estimates based on system.

    Returns:
        int: Optimal batch size
    """
    # Estimate memory if not provided
    if available_memory_gb is None:
        if torch.backends.mps.is_available() or torch.cuda.is_available():
            available_memory_gb = 16  # Conservative estimate for GPU memory
        else:
            available_memory_gb = 8   # Conservative estimate for CPU memory

    # Calculate memory requirements per sample
    bytes_per_pixel = 4  # float32
    sample_memory = image_size[0] * image_size[1] * bytes_per_pixel

    # Reserve 20% of memory for model and other operations
    usable_memory = available_memory_gb * 1e9 * 0.2

    # Calculate batch size, with a hard cap of 128
    optimal_batch_size = min(128, int(usable_memory / sample_memory))

    # Ensure batch size is at least 16
    return max(16, optimal_batch_size)

def resume_training(trainer, checkpoint_path):
    """
    Resume training from a checkpoint if available.

    Args:
        trainer (ModelTrainer): Trainer instance
        checkpoint_path (Path): Path to checkpoint file
    """
    if checkpoint_path.exists():
        trainer.load_checkpoint(checkpoint_path)
        console.print(f"[green]Resumed training from {checkpoint_path}[/green]")
        return True
    return False

def train(config: Dict):
    """Main function for training the model using simple console output."""
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/letter_classifier") / timestamp
    os.makedirs(output_dir, exist_ok=True)

    # Save current config for reproducibility (optional)
    # with open(output_dir / "config.yaml", "w") as f:
    #     yaml.dump(config, f, default_flow_style=False)

    # Calculate optimal batch size based on image dimensions
    optimal_batch_size = get_optimal_batch_size(model_config["input_shape"][:2])
    if optimal_batch_size != training_config["batch_size"]:
        console.print(
            f"[yellow]Adjusting batch size from {training_config['batch_size']} to {optimal_batch_size} based on available memory[/yellow]"
        )
        training_config["batch_size"] = optimal_batch_size

    # Create data loaders
    console.print("\n[bold cyan]Loading datasets...[/bold cyan]")
    # Assuming get_data_loaders is defined elsewhere 
    # (This assumes our data is stored in Google Drive)
    data_loaders = get_data_loaders(
        data_dir="/content/drive/MyDrive/MScProject/data/characters",
        batch_size=training_config["batch_size"],
        image_size=model_config["input_shape"][:2],
        num_workers=4,
        augment=data_config.get("augmentation", {}).get("use", True)
    )

    # Print dataset statistics
    train_size = len(data_loaders["train"].dataset)
    val_size = len(data_loaders["val"].dataset)
    test_size = len(data_loaders["test"].dataset)
    console.print(f"\n[green]Dataset Statistics:[/green]")
    console.print(f"- Training set: {train_size:,} images")
    console.print(f"- Validation set: {val_size:,} images")
    console.print(f"- Test set: {test_size:,} images")
    console.print(f"- Batch size: {training_config['batch_size']}")

    # Create model (assuming get_model is defined and imported)
    console.print("\n[bold cyan]Initializing model...[/bold cyan]")
    model = get_model(model_config["architecture"], model_config)
    console.print(f"- Input shape: {model_config['input_shape']}")
    console.print(f"- Number of classes: {model_config['num_classes']}")
    console.print(f"- Model architecture: {model_config['architecture']}")

    # Set up device
    device = setup_device()

    # Create trainer (assuming ModelTrainer is defined and imported)
    trainer = ModelTrainer(
        model=model,
        train_loader=data_loaders["train"],
        test_loader=data_loaders["val"],
        config=config,
        output_dir=output_dir,
        device=device
    )

    # Train the model
    trainer.train(training_config["epochs"])

def test_train(config: Dict):
    """Test function for training the model with a limited dataset."""
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/letter_classifier_test") / timestamp
    os.makedirs(output_dir, exist_ok=True)

    # Calculate optimal batch size based on image dimensions
    optimal_batch_size = get_optimal_batch_size(model_config["input_shape"][:2])
    if optimal_batch_size != training_config["batch_size"]:
        console.print(
            f"[yellow]Adjusting batch size from {training_config['batch_size']} to {optimal_batch_size} based on available memory[/yellow]"
        )
        training_config["batch_size"] = optimal_batch_size

    # Create data loaders with limited training set
    console.print("\n[bold cyan]Loading limited test datasets...[/bold cyan]")
    data_loaders = get_data_loaders(
        data_dir="/content/drive/MyDrive/MScProject/data/characters",
        batch_size=training_config["batch_size"],
        image_size=model_config["input_shape"][:2],
        num_workers=4,
        augment=data_config.get("augmentation", {}).get("use", True),
        limit_train_samples=20  # Limit training set to 20 images
    )

    # Print dataset statistics
    train_size = len(data_loaders["train"].dataset)
    val_size = len(data_loaders["val"].dataset)
    test_size = len(data_loaders["test"].dataset)
    console.print(f"\n[green]Dataset Statistics:[/green]")
    console.print(f"- Training set: {train_size:,} images")
    console.print(f"- Validation set: {val_size:,} images")
    console.print(f"- Test set: {test_size:,} images")
    console.print(f"- Batch size: {training_config['batch_size']}")

    # Create model
    console.print("\n[bold cyan]Initializing model...[/bold cyan]")
    model = get_model(model_config["architecture"], model_config)
    console.print(f"- Input shape: {model_config['input_shape']}")
    console.print(f"- Number of classes: {model_config['num_classes']}")
    console.print(f"- Model architecture: {model_config['architecture']}")

    # Set up device
    device = setup_device()

    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=data_loaders["train"],
        test_loader=data_loaders["val"],
        config=config,
        output_dir=output_dir,
        device=device
    )

    # Optionally resume from checkpoint
    checkpoint_path = output_dir / "latest_checkpoint.pth"
    if resume_training(trainer, checkpoint_path):
        console.print("[green]Successfully resumed training from checkpoint[/green]")

    # Train the model
    trainer.train(training_config["epochs"])
