"""
Trainer module for training the deobfuscation model.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from src.utils.config import load_config, get_training_config
from src.models.losses import get_loss_function

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for handling model training."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        output_dir: str,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Training configuration dictionary
            output_dir: Directory to save outputs (checkpoints, logs, etc.)
            device: Device to train on (CPU or GPU)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Create output directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.sample_dir = self.output_dir / 'samples'
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # Training parameters
        self.epochs = config.get('epochs', 100)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.optimizer_name = config.get('optimizer', 'adam').lower()
        self.loss_name = config.get('loss', 'bce_dice').lower()
        
        # Setup optimizer
        self.optimizer = self._get_optimizer()
        
        # Setup loss function
        self.loss_fn = get_loss_function(self.loss_name)
        
        # Setup learning rate scheduler
        self.lr_scheduler = None
        if config.get('lr_scheduler', {}).get('use', False):
            scheduler_config = config['lr_scheduler']
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-6),
                verbose=True
            )
        
        # Early stopping
        self.early_stopping = None
        if config.get('early_stopping', {}).get('use', False):
            es_config = config['early_stopping']
            self.early_stopping = EarlyStopping(
                patience=es_config.get('patience', 10),
                min_delta=es_config.get('min_delta', 0.0001),
                verbose=True
            )
        
        # Move model to device
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model: {type(model).__name__}")
        logger.info(f"Optimizer: {self.optimizer_name}")
        logger.info(f"Loss function: {self.loss_name}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _get_optimizer(self) -> optim.Optimizer:
        """
        Get optimizer based on configuration.
        
        Returns:
            PyTorch optimizer
        """
        if self.optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999)
            )
        elif self.optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=1e-4
            )
        elif self.optimizer_name == 'rmsprop':
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-5
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
    
    def train_epoch(self) -> float:
        """
        Train the model for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        # Use tqdm for progress tracking
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for batch_idx, (inputs, targets) in pbar:
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Calculate loss
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update total loss
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_description(f"Train Loss: {loss.item():.4f}")
        
        # Calculate average loss
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss
    
    def validate(self) -> Tuple[float, List[torch.Tensor]]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average validation loss, sample outputs)
        """
        self.model.eval()
        total_loss = 0.0
        samples = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.loss_fn(outputs, targets)
                
                # Update total loss
                total_loss += loss.item()
                
                # Save some samples for visualization
                if batch_idx == 0:
                    samples = [inputs[:5].cpu(), targets[:5].cpu(), outputs[:5].cpu()]
        
        # Calculate average loss
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, samples
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            logger.info(f"Epoch {epoch}/{self.epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, samples = self.validate()
            
            # Save current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
            
            # Save checkpoint
            if epoch % 5 == 0 or epoch == self.epochs:
                self.save_checkpoint(epoch)
            
            # Save sample visualizations
            if epoch % 10 == 0 or epoch == self.epochs:
                self.save_samples(epoch, samples)
            
            # Update learning rate scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(val_loss)
            
            # Check early stopping
            if self.early_stopping is not None:
                should_stop = self.early_stopping.step(val_loss)
                if should_stop:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
        
        # Calculate training time
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final model
        self.save_model('final')
        
        # Plot training history
        self.plot_history()
        
        return self.history
    
    def save_checkpoint(self, epoch: int) -> None:
        """
        Save a model checkpoint.
        
        Args:
            epoch: Current epoch
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.history['train_loss'][-1],
            'val_loss': self.history['val_loss'][-1],
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def save_model(self, name: str) -> None:
        """
        Save the model.
        
        Args:
            name: Name identifier for the saved model
        """
        model_path = self.output_dir / f"model_{name}.pth"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
    
    def save_samples(self, epoch: int, samples: List[torch.Tensor]) -> None:
        """
        Save sample visualizations.
        
        Args:
            epoch: Current epoch
            samples: List of [inputs, targets, outputs] tensors
        """
        if len(samples) != 3:
            return
        
        inputs, targets, outputs = samples
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        fig.suptitle(f'Epoch {epoch}', fontsize=16)
        
        for i in range(5):
            # Input images
            axes[0, i].imshow(inputs[i].squeeze().numpy(), cmap='gray')
            axes[0, i].set_title('Input (Obfuscated)')
            axes[0, i].axis('off')
            
            # Target images
            axes[1, i].imshow(targets[i].squeeze().numpy(), cmap='gray')
            axes[1, i].set_title('Target (Standard)')
            axes[1, i].axis('off')
            
            # Output images
            axes[2, i].imshow(outputs[i].squeeze().numpy(), cmap='gray')
            axes[2, i].set_title('Output (Deobfuscated)')
            axes[2, i].axis('off')
        
        # Save figure
        samples_path = self.sample_dir / f"samples_epoch_{epoch}.png"
        plt.tight_layout()
        plt.savefig(samples_path, dpi=200)
        plt.close()
        logger.info(f"Sample visualizations saved to {samples_path}")
    
    def plot_history(self) -> None:
        """Plot training history."""
        plt.figure(figsize=(10, 8))
        
        # Plot loss curves
        plt.subplot(2, 1, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(2, 1, 2)
        plt.plot(self.history['learning_rate'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.yscale('log')
        
        # Save figure
        history_path = self.log_dir / "training_history.png"
        plt.tight_layout()
        plt.savefig(history_path, dpi=200)
        plt.close()
        logger.info(f"Training history plot saved to {history_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Epoch number of the loaded checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Update history if available
        if 'epoch' in checkpoint and checkpoint['epoch'] > 0:
            for _ in range(checkpoint['epoch']):
                if len(self.history['train_loss']) < checkpoint['epoch']:
                    self.history['train_loss'].append(0.0)
                if len(self.history['val_loss']) < checkpoint['epoch']:
                    self.history['val_loss'].append(0.0)
                if len(self.history['learning_rate']) < checkpoint['epoch']:
                    self.history['learning_rate'].append(0.0)
            
            # Update the last epoch values
            self.history['train_loss'][-1] = checkpoint['train_loss']
            self.history['val_loss'][-1] = checkpoint['val_loss']
            self.history['learning_rate'][-1] = checkpoint['learning_rate']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get('epoch', 0)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = False):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
    
    def step(self, val_loss: float) -> bool:
        """
        Update early stopping state.
        
        Args:
            val_loss: Validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_loss
            return False
        
        if val_loss < self.best_score - self.min_delta:
            # Performance improved
            self.best_score = val_loss
            self.counter = 0
            return False
        
        # Performance did not improve
        self.counter += 1
        if self.verbose:
            logger.info(f"Early stopping counter: {self.counter}/{self.patience}")
        
        if self.counter >= self.patience:
            if self.verbose:
                logger.info("Early stopping triggered")
            return True
        
        return False
    