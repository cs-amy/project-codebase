"""
Script for training the letter classification model.
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
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import load_config, get_model_config, get_training_config, get_data_config
from data.data_loader import get_data_loaders
from models.letter_classifier import get_model
from trainer import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    
    # Reserve 20% of memory for the model and other operations
    usable_memory = available_memory_gb * 1e9 * 0.2
    
    # Calculate batch size
    optimal_batch_size = min(128, int(usable_memory / sample_memory))
    
    # Ensure batch size is at least 16
    return max(16, optimal_batch_size)

def resume_training(trainer, checkpoint_path):
    """
    Resume training from a checkpoint if available.
    
    Args:
        trainer (Trainer): Training instance
        checkpoint_path (Path): Path to checkpoint file
    """
    if checkpoint_path.exists():
        trainer.load_checkpoint(checkpoint_path)
        console.print(f"[green]Resumed training from {checkpoint_path}[/green]")
        return True
    return False

def monitor_memory():
    """
    Monitor and log GPU/CPU memory usage during training.
    Returns a formatted string with memory information.
    """
    memory_info = []
    
    if torch.backends.mps.is_available():
        try:
            used_memory = torch.mps.current_allocated_memory() / 1e9
            memory_info.append(f"GPU Memory Used: {used_memory:.2f} GB")
        except:
            memory_info.append("GPU Memory: Not available")
    elif torch.cuda.is_available():
        try:
            used_memory = torch.cuda.memory_allocated() / 1e9
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            memory_info.append(f"GPU Memory: {used_memory:.2f}GB / {total_memory:.2f}GB")
        except:
            memory_info.append("GPU Memory: Not available")
    else:
        import psutil
        process = psutil.Process()
        used_memory = process.memory_info().rss / 1e9
        total_memory = psutil.virtual_memory().total / 1e9
        memory_info.append(f"CPU Memory: {used_memory:.2f}GB / {total_memory:.2f}GB")
    
    return " | ".join(memory_info)

def create_layout():
    """Create the layout for the training display."""
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )
    return layout

def create_header():
    """Create the header panel with training information."""
    return Panel(
        "[bold blue]Letter Classification Model Training[/bold blue]",
        style="white on blue"
    )

def create_footer(epoch, total_epochs, train_loss, train_acc, val_loss, val_acc, lr):
    """Create the footer panel with current training metrics."""
    return Panel(
        f"Epoch: {epoch}/{total_epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}% | "
        f"LR: {lr:.6f}",
        style="white on blue"
    )

def create_metrics_table(train_loss, train_acc, val_loss, val_acc):
    """Create a table with training metrics."""
    table = Table(title="Training Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Training Loss", f"{train_loss:.4f}")
    table.add_row("Training Accuracy", f"{train_acc:.2f}%")
    table.add_row("Validation Loss", f"{val_loss:.4f}")
    table.add_row("Validation Accuracy", f"{val_acc:.2f}%")
    
    return table

def main():
    """Main function for training the model."""
    # Load config
    config_path = Path("configs/train_config.yaml")
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        return
    
    config = load_config(config_path)
    model_config = get_model_config(config)
    training_config = get_training_config(config)
    data_config = get_data_config(config)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/letter_classifier") / timestamp
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Calculate optimal batch size
    optimal_batch_size = get_optimal_batch_size(model_config["input_shape"][:2])
    if optimal_batch_size != training_config["batch_size"]:
        console.print(f"[yellow]Adjusting batch size from {training_config['batch_size']} to {optimal_batch_size} based on available memory[/yellow]")
        training_config["batch_size"] = optimal_batch_size
    
    # Create data loaders
    console.print("\n[bold cyan]Loading datasets...[/bold cyan]")
    data_loaders = get_data_loaders(
        data_dir="data/characters",
        batch_size=training_config["batch_size"],
        image_size=model_config["input_shape"][:2],
        num_workers=4,
        augment=data_config.get("augmentation", {}).get("use", True)
    )
    
    # Print dataset statistics
    train_size = len(data_loaders["train"].dataset)
    val_size = len(data_loaders["val"].dataset)
    console.print(f"\n[green]Dataset Statistics:[/green]")
    console.print(f"- Training set: {train_size:,} images")
    console.print(f"- Validation set: {val_size:,} images")
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
    trainer = Trainer(
        model=model,
        train_loader=data_loaders["train"],
        val_loader=data_loaders["val"],
        config=training_config,
        output_dir=output_dir,
        device=device
    )
    
    # Try to resume from checkpoint
    checkpoint_path = output_dir / "latest_checkpoint.pth"
    if resume_training(trainer, checkpoint_path):
        console.print("[green]Successfully resumed training from checkpoint[/green]")
    
    # Create layout for training display
    layout = create_layout()
    
    # Start training with rich display
    console.print("\n[bold cyan]Starting training...[/bold cyan]")
    with Live(layout, refresh_per_second=4) as live:
        for epoch in range(1, training_config["epochs"] + 1):
            # Update header
            layout["header"].update(create_header())
            
            # Monitor and display memory usage
            memory_status = monitor_memory()
            console.print(f"\n[cyan]Memory Status: {memory_status}[/cyan]")
            
            # Train for one epoch
            train_loss, train_acc = trainer.train_epoch()
            
            # Validate
            val_loss, val_acc, predictions, targets = trainer.validate()
            
            # Update footer with current metrics
            layout["footer"].update(create_footer(
                epoch, training_config["epochs"],
                train_loss, train_acc,
                val_loss, val_acc,
                trainer.optimizer.param_groups[0]['lr']
            ))
            
            # Update metrics table
            layout["body"].update(create_metrics_table(
                train_loss, train_acc,
                val_loss, val_acc
            ))
            
            # Save checkpoint and visualizations
            if epoch % 5 == 0:
                trainer.save_checkpoint(epoch)
                # Monitor memory after checkpoint save
                memory_status = monitor_memory()
                console.print(f"[cyan]Memory Status after checkpoint: {memory_status}[/cyan]")
            
            if epoch % 10 == 0:
                trainer.plot_confusion_matrix(predictions, targets, epoch)
    
    # Save final model and plots
    trainer.save_model("final")
    trainer.plot_history()
    
    # Final memory status
    memory_status = monitor_memory()
    console.print(f"\n[cyan]Final Memory Status: {memory_status}[/cyan]")
    
    console.print("\n[bold green]Training completed![/bold green]")
    console.print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
