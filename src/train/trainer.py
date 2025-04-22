"""
Trainer module for setting up and running the letter classification model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from rich.console import Console

from src.models.letter_classifier import LetterClassifierCNN
from src.data.data_loader import CharacterDataset

console = Console()

class ModelTrainer:
    """Trainer class for the character classification model."""
    
    def __init__(
        self,
        model: LetterClassifierCNN,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        output_dir: str | Path,
        device: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            config: Training configuration dictionary
            output_dir: Directory to save checkpoints and results
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )
        
        # Initialize learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['training']['lr_scheduler']['factor'],
            patience=config['training']['lr_scheduler']['patience'],
            min_lr=config['training']['lr_scheduler']['min_lr']
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_model_path = self.output_dir / 'best_model.pth'
        self.patience_counter = 0
        self.early_stopping_patience = config['training']['early_stopping']['patience']
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Get character mappings
        self.char_to_idx, self.idx_to_char = CharacterDataset.get_char_mapping()
        
        console.print(f"[green]Initialized trainer on device: {self.device}[/green]")
        console.print(f"[green]Output directory: {self.output_dir}[/green]")
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Validating'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save a checkpoint of the model.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if applicable
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            console.print(f"[green]Saved best model to {self.best_model_path}[/green]")
    
    def plot_training_history(self) -> None:
        """Plot and save training history."""
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['val_acc'], label='Val Acc')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png')
        plt.show()
    
    def plot_confusion_matrix(self) -> None:
        """Plot and save confusion matrix with character labels."""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                all_preds.extend(predicted.cpu().detach().numpy())
                all_targets.extend(target.cpu().detach().numpy())
        
        cm = confusion_matrix(all_targets, all_preds)
        
        # Create figure with larger size
        plt.figure(figsize=(15, 12))
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[self.idx_to_char[i] for i in range(26)],
            yticklabels=[self.idx_to_char[i] for i in range(26)]
        )
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Character')
        plt.ylabel('True Character')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save figure with high DPI
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Log per-character accuracy
        char_acc = cm.diagonal() / cm.sum(axis=1)
        console.print("\nPer-character accuracy:")
        for i, acc in enumerate(char_acc):
            console.print(f"{self.idx_to_char[i]}: {acc*100:.2f}%")
    
    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        """
        Load a checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            The epoch number of the loaded checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        console.print(f"[green]Loaded checkpoint from {checkpoint_path}[/green]")
        console.print(f"[green]Resuming from epoch {checkpoint['epoch']}[/green]")
        
        return checkpoint['epoch']
    
    def train(self, num_epochs: int) -> None:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train for
        """
        console.print("[green]Starting training...[/green]")
        
        for epoch in range(1, num_epochs + 1):
            console.print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            console.print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            console.print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            console.print(f"Learning Rate: {current_lr:.6f}")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Save checkpoint if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, is_best)
            
            # Plot confusion matrix every 10 epochs
            # if epoch % 10 == 0:
                # self.plot_confusion_matrix()
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                console.print(f"[green]Early stopping triggered after {epoch} epochs[/green]")
                break
        
        # Save final plots and checkpoint
        self.plot_training_history()
        # self.plot_confusion_matrix()
        self.save_checkpoint(num_epochs)
        
        console.print("[green]Training completed![/green]")
