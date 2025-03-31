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
    output_dir = Path("outputs/letter_classifier")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
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
    
    # Create model
    console.print("\n[bold cyan]Initializing model...[/bold cyan]")
    model = get_model(model_config["architecture"], model_config)
    console.print(f"- Input shape: {model_config['input_shape']}")
    console.print(f"- Number of classes: {model_config['num_classes']}")
    console.print(f"- Model architecture: {model_config['architecture']}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"\n[green]Using device: {device}[/green]")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=data_loaders["train"],
        val_loader=data_loaders["val"],
        config=training_config,
        output_dir=output_dir,
        device=device
    )
    
    # Create layout for training display
    layout = create_layout()
    
    # Start training with rich display
    console.print("\n[bold cyan]Starting training...[/bold cyan]")
    with Live(layout, refresh_per_second=4) as live:
        for epoch in range(1, training_config["epochs"] + 1):
            # Update header
            layout["header"].update(create_header())
            
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
            
            if epoch % 10 == 0:
                trainer.plot_confusion_matrix(predictions, targets, epoch)
    
    # Save final model and plots
    trainer.save_model("final")
    trainer.plot_history()
    
    console.print("\n[bold green]Training completed![/bold green]")
    console.print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
