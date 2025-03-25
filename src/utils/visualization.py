#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for displaying and saving results of the deobfuscation model and OCR system.

This module provides functions to create visualizations of model inputs, outputs, and performance
metrics to help understand and evaluate the deobfuscation process.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def plot_image_pairs(input_images: List[np.ndarray], 
                    output_images: List[np.ndarray],
                    target_images: Optional[List[np.ndarray]] = None,
                    titles: Optional[List[str]] = None,
                    ocr_results: Optional[List[Dict]] = None,
                    max_samples: int = 10,
                    save_path: Optional[str] = None):
    """
    Plot pairs of input and output images, optionally with target images and OCR results.
    
    Args:
        input_images (List[np.ndarray]): List of input (obfuscated) images.
        output_images (List[np.ndarray]): List of output (deobfuscated) images.
        target_images (Optional[List[np.ndarray]], optional): List of target (ground truth) images. 
                                                            Defaults to None.
        titles (Optional[List[str]], optional): List of titles for each row. Defaults to None.
        ocr_results (Optional[List[Dict]], optional): List of OCR results for each image set. 
                                                    Each dict should have 'input', 'output', and optionally 'target' keys.
                                                    Defaults to None.
        max_samples (int, optional): Maximum number of samples to display. Defaults to 10.
        save_path (Optional[str], optional): Path to save the plot. If None, display instead. Defaults to None.
    """
    # Limit the number of samples
    n_samples = min(len(input_images), len(output_images), max_samples)
    
    # Determine number of columns
    n_cols = 2  # Input and output
    if target_images is not None:
        n_cols = 3  # Input, output, and target
    
    # Create figure
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(n_cols * 3, n_samples * 3))
    
    # Handle case with single sample
    if n_samples == 1:
        axes = np.array([axes])
    
    # Plot each sample
    for i in range(n_samples):
        # Get images
        input_img = input_images[i]
        output_img = output_images[i]
        
        # Ensure images are in the right format for display
        if len(input_img.shape) == 3 and input_img.shape[2] == 1:
            input_img = input_img[:, :, 0]
        if len(output_img.shape) == 3 and output_img.shape[2] == 1:
            output_img = output_img[:, :, 0]
        
        # Plot input image
        axes[i, 0].imshow(input_img, cmap='gray')
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        
        # Plot output image
        axes[i, 1].imshow(output_img, cmap='gray')
        axes[i, 1].set_title('Output')
        axes[i, 1].axis('off')
        
        # Plot target image if provided
        if target_images is not None and i < len(target_images):
            target_img = target_images[i]
            if len(target_img.shape) == 3 and target_img.shape[2] == 1:
                target_img = target_img[:, :, 0]
            axes[i, 2].imshow(target_img, cmap='gray')
            axes[i, 2].set_title('Target')
            axes[i, 2].axis('off')
        
        # Add OCR results if provided
        if ocr_results is not None and i < len(ocr_results):
            result = ocr_results[i]
            if 'input' in result:
                axes[i, 0].set_xlabel(f"OCR: {result['input']}")
            if 'output' in result:
                axes[i, 1].set_xlabel(f"OCR: {result['output']}")
            if 'target' in result and target_images is not None:
                axes[i, 2].set_xlabel(f"OCR: {result['target']}")
        
        # Add row title if provided
        if titles is not None and i < len(titles):
            axes[i, 0].set_ylabel(titles[i], rotation=0, labelpad=40, va='center', fontsize=12)
    
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
        plt.close(fig)
    else:
        plt.show()

def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    Plot training history metrics.
    
    Args:
        history (Dict): Dictionary containing training history (loss, metrics, etc.).
                       Expected to have keys like 'loss', 'val_loss', etc.
        save_path (Optional[str], optional): Path to save the plot. If None, display instead. Defaults to None.
    """
    # Create figure with subplots for each metric
    metrics = [k for k in history.keys() if not k.startswith('val_')]
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)
    
    # Handle case with single metric
    if n_metrics == 1:
        axes = np.array([axes])
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(history[metric], label=f'Training {metric}')
        
        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Validation {metric}')
        
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Set x-axis label on the bottom plot
    axes[-1].set_xlabel('Epochs')
    
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()

def plot_confusion_matrix(confusion_matrix: np.ndarray, 
                          class_names: List[str],
                          title: str = 'Confusion Matrix',
                          cmap: str = 'Blues',
                          normalize: bool = True,
                          save_path: Optional[str] = None):
    """
    Plot a confusion matrix.
    
    Args:
        confusion_matrix (np.ndarray): The confusion matrix to plot.
        class_names (List[str]): List of class names for the axis labels.
        title (str, optional): Title of the plot. Defaults to 'Confusion Matrix'.
        cmap (str, optional): Colormap to use. Defaults to 'Blues'.
        normalize (bool, optional): Whether to normalize the confusion matrix. Defaults to True.
        save_path (Optional[str], optional): Path to save the plot. If None, display instead. Defaults to None.
    """
    # Normalize the confusion matrix if requested
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-6)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Set up the figure
    plt.figure(figsize=(10, 8))
    
    # Plot the confusion matrix
    sns.heatmap(confusion_matrix, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1 if normalize else None)
    
    plt.title(title)
    plt.ylabel('True Character')
    plt.xlabel('Predicted Character')
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_ocr_accuracy_comparison(original_accuracies: List[float],
                                processed_accuracies: List[float],
                                bins: int = 10,
                                title: str = 'OCR Accuracy Improvement',
                                save_path: Optional[str] = None):
    """
    Plot a histogram of OCR accuracy before and after processing.
    
    Args:
        original_accuracies (List[float]): List of OCR accuracies on original images.
        processed_accuracies (List[float]): List of OCR accuracies on processed images.
        bins (int, optional): Number of histogram bins. Defaults to 10.
        title (str, optional): Title of the plot. Defaults to 'OCR Accuracy Improvement'.
        save_path (Optional[str], optional): Path to save the plot. If None, display instead. Defaults to None.
    """
    # Calculate improvement
    improvements = [processed - original for processed, original in zip(processed_accuracies, original_accuracies)]
    mean_improvement = np.mean(improvements)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original accuracies
    ax1.hist(original_accuracies, bins=bins, alpha=0.7, color='red')
    ax1.set_title('Original OCR Accuracy')
    ax1.set_xlabel('Accuracy')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(0, 1)
    ax1.axvline(np.mean(original_accuracies), color='black', linestyle='dashed', linewidth=1)
    ax1.text(0.05, 0.95, f'Mean: {np.mean(original_accuracies):.3f}', 
             transform=ax1.transAxes, va='top')
    
    # Plot processed accuracies
    ax2.hist(processed_accuracies, bins=bins, alpha=0.7, color='green')
    ax2.set_title('Processed OCR Accuracy')
    ax2.set_xlabel('Accuracy')
    ax2.set_xlim(0, 1)
    ax2.axvline(np.mean(processed_accuracies), color='black', linestyle='dashed', linewidth=1)
    ax2.text(0.05, 0.95, f'Mean: {np.mean(processed_accuracies):.3f}', 
             transform=ax2.transAxes, va='top')
    
    # Plot improvement
    ax3.hist(improvements, bins=bins, alpha=0.7, color='blue')
    ax3.set_title('Accuracy Improvement')
    ax3.set_xlabel('Improvement')
    ax3.axvline(mean_improvement, color='black', linestyle='dashed', linewidth=1)
    ax3.text(0.05, 0.95, f'Mean: {mean_improvement:.3f}', 
             transform=ax3.transAxes, va='top')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved OCR accuracy comparison to {save_path}")
        plt.close(fig)
    else:
        plt.show()

def create_side_by_side_image(input_image: np.ndarray,
                             output_image: np.ndarray,
                             input_text: Optional[str] = None,
                             output_text: Optional[str] = None,
                             conf_input: Optional[float] = None,
                             conf_output: Optional[float] = None) -> np.ndarray:
    """
    Create a side-by-side comparison image of input and output.
    
    Args:
        input_image (np.ndarray): Input (obfuscated) image.
        output_image (np.ndarray): Output (deobfuscated) image.
        input_text (Optional[str], optional): OCR text for input image. Defaults to None.
        output_text (Optional[str], optional): OCR text for output image. Defaults to None.
        conf_input (Optional[float], optional): OCR confidence for input. Defaults to None.
        conf_output (Optional[float], optional): OCR confidence for output. Defaults to None.
        
    Returns:
        np.ndarray: Side-by-side comparison image.
    """
    # Ensure images are 3-channel (needed for OpenCV drawing)
    if len(input_image.shape) == 2 or input_image.shape[2] == 1:
        input_rgb = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    else:
        input_rgb = input_image.copy()
    
    if len(output_image.shape) == 2 or output_image.shape[2] == 1:
        output_rgb = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    else:
        output_rgb = output_image.copy()
    
    # Get image dimensions
    h_in, w_in = input_rgb.shape[:2]
    h_out, w_out = output_rgb.shape[:2]
    
    # Create labels
    input_label = "Input"
    output_label = "Output"
    
    if input_text is not None:
        input_label += f" (OCR: '{input_text}'"
        if conf_input is not None:
            input_label += f", conf: {conf_input:.2f}"
        input_label += ")"
    
    if output_text is not None:
        output_label += f" (OCR: '{output_text}'"
        if conf_output is not None:
            output_label += f", conf: {conf_output:.2f}"
        output_label += ")"
    
    # Add label space
    label_height = 30
    input_with_label = np.ones((h_in + label_height, w_in, 3), dtype=input_rgb.dtype) * 255
    output_with_label = np.ones((h_out + label_height, w_out, 3), dtype=output_rgb.dtype) * 255
    
    input_with_label[label_height:, :, :] = input_rgb
    output_with_label[label_height:, :, :] = output_rgb
    
    # Add labels
    cv2.putText(input_with_label, input_label, (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(output_with_label, output_label, (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Create side-by-side image
    if h_in + label_height == h_out + label_height and w_in == w_out:
        # Images are the same size, simple concatenation
        side_by_side = np.hstack((input_with_label, output_with_label))
    else:
        # Images are different sizes, need to resize
        max_h = max(h_in, h_out) + label_height
        max_w = max(w_in, w_out)
        
        resized_input = np.ones((max_h, max_w, 3), dtype=input_rgb.dtype) * 255
        resized_output = np.ones((max_h, max_w, 3), dtype=output_rgb.dtype) * 255
        
        resized_input[label_height:label_height + h_in, :w_in, :] = input_rgb
        resized_output[label_height:label_height + h_out, :w_out, :] = output_rgb
        
        # Add labels to the resized images
        cv2.putText(resized_input, input_label, (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(resized_output, output_label, (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        side_by_side = np.hstack((resized_input, resized_output))
    
    return side_by_side

def save_side_by_side_image(input_image: np.ndarray,
                           output_image: np.ndarray,
                           save_path: str,
                           input_text: Optional[str] = None,
                           output_text: Optional[str] = None,
                           conf_input: Optional[float] = None,
                           conf_output: Optional[float] = None):
    """
    Create and save a side-by-side comparison image of input and output.
    
    Args:
        input_image (np.ndarray): Input (obfuscated) image.
        output_image (np.ndarray): Output (deobfuscated) image.
        save_path (str): Path to save the image.
        input_text (Optional[str], optional): OCR text for input image. Defaults to None.
        output_text (Optional[str], optional): OCR text for output image. Defaults to None.
        conf_input (Optional[float], optional): OCR confidence for input. Defaults to None.
        conf_output (Optional[float], optional): OCR confidence for output. Defaults to None.
    """
    side_by_side = create_side_by_side_image(
        input_image, output_image, input_text, output_text, conf_input, conf_output
    )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # Save the image
    cv2.imwrite(save_path, side_by_side)
    logger.info(f"Saved side-by-side comparison to {save_path}")

if __name__ == "__main__":
    # Simple demo code
    logging.basicConfig(level=logging.INFO)
    
    # Create sample images
    np.random.seed(42)
    input_images = [np.random.rand(64, 64) for _ in range(5)]
    output_images = [np.random.rand(64, 64) for _ in range(5)]
    target_images = [np.random.rand(64, 64) for _ in range(5)]
    
    # Create sample OCR results
    ocr_results = [
        {'input': 'a', 'output': 'a', 'target': 'a'},
        {'input': '@', 'output': 'a', 'target': 'a'},
        {'input': 'b', 'output': 'b', 'target': 'b'},
        {'input': '8', 'output': 'B', 'target': 'B'},
        {'input': '0', 'output': 'o', 'target': 'o'}
    ]
    
    # Plot image pairs
    plot_image_pairs(
        input_images, output_images, target_images,
        titles=['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5'],
        ocr_results=ocr_results,
        save_path='demo_image_pairs.png'
    )
    
    # Create sample training history
    history = {
        'loss': [0.5, 0.4, 0.3, 0.25, 0.2],
        'val_loss': [0.55, 0.45, 0.35, 0.3, 0.25],
        'accuracy': [0.8, 0.85, 0.9, 0.92, 0.95],
        'val_accuracy': [0.75, 0.8, 0.85, 0.87, 0.9]
    }
    
    # Plot training history
    plot_training_history(history, save_path='demo_training_history.png')
    
    # Create sample confusion matrix
    confusion_matrix = np.array([
        [0.8, 0.1, 0.1],
        [0.05, 0.9, 0.05],
        [0.1, 0.1, 0.8]
    ])
    class_names = ['a', 'b', 'c']
    
    # Plot confusion matrix
    plot_confusion_matrix(
        confusion_matrix, class_names,
        title='Demo Confusion Matrix',
        save_path='demo_confusion_matrix.png'
    )
    
    # Create sample OCR accuracies
    original_accuracies = [0.6, 0.7, 0.5, 0.6, 0.8, 0.7, 0.6, 0.5, 0.7, 0.6]
    processed_accuracies = [0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.8, 0.7, 0.9, 0.8]
    
    # Plot OCR accuracy comparison
    plot_ocr_accuracy_comparison(
        original_accuracies, processed_accuracies,
        title='Demo OCR Accuracy Improvement',
        save_path='demo_ocr_accuracy_comparison.png'
    )
    
    logger.info("Demo visualization complete") 