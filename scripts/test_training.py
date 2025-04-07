"""
Script to run a small test training to verify the training pipeline.
"""

import os
import sys
from pathlib import Path
import shutil
import random
import torch
import yaml
from datetime import datetime
from rich.console import Console

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.train.train_letter_classifier import main as train_main
from utils.config import load_config

console = Console()

def create_test_dataset(source_dir: Path, dest_dir: Path, samples_per_class: int = 100):
    """Create a small test dataset by copying a subset of images."""
    console.print(f"\n[bold cyan]Creating test dataset from {source_dir}...[/bold cyan]")
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy a subset of images for each character
    total_copied = 0
    for char_dir in source_dir.iterdir():
        if char_dir.is_dir():
            dest_char_dir = dest_dir / char_dir.name
            os.makedirs(dest_char_dir, exist_ok=True)
            
            # Get list of image files
            image_files = list(char_dir.glob("*.png"))
            if not image_files:
                console.print(f"[yellow]Warning: No PNG files found in {char_dir}[/yellow]")
                continue
            
            # Select random subset
            selected_files = random.sample(
                image_files, 
                min(samples_per_class, len(image_files))
            )
            
            # Copy selected files
            for img_file in selected_files:
                shutil.copy2(img_file, dest_char_dir / img_file.name)
                total_copied += 1
    
    console.print(f"[green]Created test dataset with {total_copied} total images[/green]")

def modify_config_for_test(config):
    """Modify the config for quick testing."""
    # Create a deep copy to avoid modifying the original
    test_config = config.copy()
    
    # Modify training parameters
    test_config["training"].update({
        "epochs": 5,
        "batch_size": 32
    })
    test_config["training"]["early_stopping"]["patience"] = 3
    test_config["training"]["scheduler"]["patience"] = 2
    
    # Update paths for test dataset
    test_config["data"].update({
        "regular": {
            "train_dir": "data/test_characters/train/regular",
            "test_dir": "data/test_characters/test/regular"
        },
        "obfuscated": {
            "train_dir": "data/test_characters/train/obfuscated",
            "test_dir": "data/test_characters/test/obfuscated"
        }
    })
    
    # Update output directory for test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_config["output"]["dir"] = f"outputs/test_runs/{timestamp}"
    
    return test_config

def setup_test_environment():
    """Set up the test environment including directories."""
    # Create test directories
    test_dirs = {
        "train": {
            "regular": Path("data/test_characters/train/regular"),
            "obfuscated": Path("data/test_characters/train/obfuscated")
        },
        "test": {
            "regular": Path("data/test_characters/test/regular"),
            "obfuscated": Path("data/test_characters/test/obfuscated")
        }
    }
    
    for split_dirs in test_dirs.values():
        for dir_path in split_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    return test_dirs

def main():
    """Run a test training with a small dataset."""
    console.print("[bold]Starting Test Training Setup[/bold]")
    
    # Load original config
    config_path = Path("configs/train_config.yaml")
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        return
    
    config = load_config(config_path)
    
    # Set up test environment
    test_dirs = setup_test_environment()
    
    # Create test datasets
    try:
        # Create training sets
        create_test_dataset(
            Path(config["data"]["regular"]["train_dir"]),
            test_dirs["train"]["regular"],
            samples_per_class=100
        )
        create_test_dataset(
            Path(config["data"]["obfuscated"]["train_dir"]),
            test_dirs["train"]["obfuscated"],
            samples_per_class=100
        )
        
        # Create smaller test sets
        create_test_dataset(
            Path(config["data"]["regular"]["test_dir"]),
            test_dirs["test"]["regular"],
            samples_per_class=20
        )
        create_test_dataset(
            Path(config["data"]["obfuscated"]["test_dir"]),
            test_dirs["test"]["obfuscated"],
            samples_per_class=20
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error creating test dataset: {str(e)}[/red]")
        return
    
    # Modify config for testing
    test_config = modify_config_for_test(config)
    
    # Save test config
    test_config_path = Path("configs/test_config.yaml")
    with open(test_config_path, "w") as f:
        yaml.dump(test_config, f, default_flow_style=False)
    
    # Run test training
    console.print("\n[bold cyan]Starting test training...[/bold cyan]")
    try:
        train_main(test_config)
        console.print("\n[bold green]Test training completed successfully![/bold green]")
        console.print(f"Results saved to: {test_config['output']['dir']}")
    except Exception as e:
        console.print(f"\n[bold red]Test training failed with error:[/bold red]")
        console.print(f"[red]{str(e)}[/red]")
        raise

if __name__ == "__main__":
    main()
