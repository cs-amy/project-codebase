"""
Evaluator module for testing and evaluating the deobfuscation model.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
from typing import Dict, List, Optional, Any
import logging
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.utils.evaluation import (
    pixel_accuracy,
    dice_coefficient,
    iou_score,
    structural_similarity_index,
    calculate_all_metrics
)
from src.utils.ocr import (
    recognize_single_character,
    recognize_word,
    compare_ocr_results
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator class for testing and evaluating the deobfuscation model."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        output_dir: Optional[str] = None,
        threshold: float = 0.5
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: PyTorch model to evaluate
            test_loader: DataLoader for test data
            device: Device to evaluate on (CPU or GPU)
            output_dir: Directory to save evaluation results
            threshold: Threshold for binarizing predictions
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.threshold = threshold
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Get dataset mapping if available
        self.mapping = {}
        try:
            if hasattr(test_loader.dataset, 'mapping'):
                self.mapping = test_loader.dataset.mapping
        except:
            logger.warning("Could not retrieve character mapping from dataset")
        
        logger.info(f"Evaluator initialized on device: {self.device}")
    
    def evaluate(
        self,
        save_results: bool = True,
        save_samples: bool = True,
        num_samples: int = 10,
        ocr_evaluation: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the model on the test set.
        
        Args:
            save_results: Whether to save evaluation results
            save_samples: Whether to save sample visualizations
            num_samples: Number of sample visualizations to save
            ocr_evaluation: Whether to perform OCR evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting evaluation...")
        
        # Lists to store evaluation results
        all_metrics = []
        all_preds = []
        all_targets = []
        all_inputs = []
        ocr_results = [] if ocr_evaluation else None
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.test_loader)):
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Move tensors to CPU for evaluation
                inputs_np = inputs.cpu().numpy()
                outputs_np = outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                # Store for later use
                all_inputs.extend([img.squeeze() for img in inputs_np])
                all_preds.extend([img.squeeze() for img in outputs_np])
                all_targets.extend([img.squeeze() for img in targets_np])
                
                # Calculate metrics for each image in the batch
                for i in range(inputs.size(0)):
                    pred = outputs_np[i].squeeze()
                    target = targets_np[i].squeeze()
                    
                    # Calculate metrics
                    metrics = calculate_all_metrics(target, pred, threshold=self.threshold)
                    all_metrics.append(metrics)
                
                # Perform OCR evaluation if requested
                if ocr_evaluation and batch_idx < 5:  # Limit OCR evaluation to first 5 batches for speed
                    self._evaluate_ocr(inputs_np, outputs_np, targets_np, ocr_results)
        
        # Calculate average metrics
        avg_metrics = self._calculate_average_metrics(all_metrics)
        
        # Compile results
        results = {
            'metrics': avg_metrics,
            'ocr_results': self._calculate_ocr_metrics(ocr_results) if ocr_evaluation else None,
        }
        
        # Save visualizations
        if save_samples:
            self._save_sample_visualizations(all_inputs, all_preds, all_targets, num_samples)
        
        # Save results
        if save_results:
            self._save_results(results)
        
        logger.info("Evaluation completed")
        logger.info(f"Average metrics: {json.dumps(avg_metrics, indent=2)}")
        
        return results
    
    def _evaluate_ocr(
        self,
        inputs_np: np.ndarray,
        outputs_np: np.ndarray,
        targets_np: np.ndarray,
        ocr_results: List
    ) -> None:
        """
        Evaluate OCR performance.
        
        Args:
            inputs_np: Input images
            outputs_np: Model outputs
            targets_np: Target images
            ocr_results: List to store OCR results
        """
        for i in range(len(inputs_np)):
            # Convert to uint8 for OCR
            input_img = (inputs_np[i].squeeze() * 255).astype(np.uint8)
            output_img = (outputs_np[i].squeeze() * 255).astype(np.uint8)
            target_img = (targets_np[i].squeeze() * 255).astype(np.uint8)
            
            # Apply thresholding
            _, output_bin = cv2.threshold(output_img, int(self.threshold * 255), 255, cv2.THRESH_BINARY)
            
            # Get true character if mapping is available
            true_char = None
            if self.mapping:
                img_filename = self.test_loader.dataset.image_files[i]
                if img_filename in self.mapping:
                    true_char = self.mapping[img_filename].get('standard_char')
            
            # Perform OCR
            input_char = recognize_single_character(input_img)
            output_char = recognize_single_character(output_bin)
            target_char = recognize_single_character(target_img)
            
            # Store results
            ocr_result = {
                'input_ocr': input_char,
                'output_ocr': output_char,
                'target_ocr': target_char,
                'true_char': true_char,
                'input_correct': input_char == true_char if true_char else None,
                'output_correct': output_char == true_char if true_char else None,
                'target_correct': target_char == true_char if true_char else None
            }
            ocr_results.append(ocr_result)
    
    def _calculate_average_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate average metrics.
        
        Args:
            all_metrics: List of metric dictionaries for individual images
            
        Returns:
            Dictionary of average metrics
        """
        avg_metrics = {}
        
        if not all_metrics:
            return avg_metrics
        
        # Initialize with keys from the first dictionary
        for key in all_metrics[0].keys():
            avg_metrics[key] = 0.0
        
        # Sum all values
        for metrics in all_metrics:
            for key, value in metrics.items():
                avg_metrics[key] += value
        
        # Calculate average
        num_samples = len(all_metrics)
        for key in avg_metrics.keys():
            avg_metrics[key] /= num_samples
        
        return avg_metrics
    
    def _calculate_ocr_metrics(self, ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate OCR-related metrics.
        
        Args:
            ocr_results: List of OCR results
            
        Returns:
            Dictionary of OCR metrics
        """
        if not ocr_results:
            return {}
        
        # Count samples with valid true_char
        valid_samples = [r for r in ocr_results if r['true_char'] is not None]
        if not valid_samples:
            return {
                'input_accuracy': None,
                'output_accuracy': None,
                'target_accuracy': None,
                'improvement': None
            }
        
        # Calculate accuracy
        input_correct = sum(1 for r in valid_samples if r['input_correct'])
        output_correct = sum(1 for r in valid_samples if r['output_correct'])
        target_correct = sum(1 for r in valid_samples if r['target_correct'])
        
        num_valid = len(valid_samples)
        input_acc = input_correct / num_valid
        output_acc = output_correct / num_valid
        target_acc = target_correct / num_valid
        
        # Calculate improvement
        improvement = (output_acc - input_acc) / max(input_acc, 1e-8)
        
        return {
            'input_accuracy': input_acc,
            'output_accuracy': output_acc,
            'target_accuracy': target_acc,
            'improvement': improvement,
            'num_samples': num_valid
        }
    
    def _save_sample_visualizations(
        self,
        inputs: List[np.ndarray],
        outputs: List[np.ndarray],
        targets: List[np.ndarray],
        num_samples: int
    ) -> None:
        """
        Save sample visualizations.
        
        Args:
            inputs: List of input images
            outputs: List of output images
            targets: List of target images
            num_samples: Number of samples to visualize
        """
        num_samples = min(num_samples, len(inputs))
        if num_samples == 0:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
        fig.suptitle('Evaluation Samples', fontsize=16)
        
        for i in range(num_samples):
            # Input images
            axes[0, i].imshow(inputs[i], cmap='gray')
            axes[0, i].set_title('Input (Obfuscated)')
            axes[0, i].axis('off')
            
            # Output images
            axes[1, i].imshow(outputs[i], cmap='gray')
            axes[1, i].set_title('Output (Deobfuscated)')
            axes[1, i].axis('off')
            
            # Target images
            axes[2, i].imshow(targets[i], cmap='gray')
            axes[2, i].set_title('Target (Standard)')
            axes[2, i].axis('off')
        
        # Save figure
        samples_path = self.output_dir / "sample_visualizations.png"
        plt.tight_layout()
        plt.savefig(samples_path, dpi=200)
        plt.close()
        logger.info(f"Sample visualizations saved to {samples_path}")
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save evaluation results.
        
        Args:
            results: Evaluation results
        """
        # Save metrics as JSON
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        # Save OCR results if available
        if results['ocr_results']:
            ocr_path = self.output_dir / "ocr_metrics.json"
            with open(ocr_path, 'w') as f:
                json.dump(results['ocr_results'], f, indent=2)
        
        logger.info(f"Evaluation results saved to {self.output_dir}")
    
    def evaluate_pipeline_comparison(
        self,
        test_images: List[np.ndarray],
        true_characters: List[str],
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Compare OCR pipeline with and without deobfuscation.
        
        Args:
            test_images: List of test images
            true_characters: List of true characters
            save_results: Whether to save evaluation results
            
        Returns:
            Dictionary of comparison results
        """
        logger.info("Comparing OCR pipelines (with vs. without deobfuscation)...")
        
        results = {
            'direct_ocr': [],
            'deobfuscated_ocr': [],
            'metrics': {}
        }
        
        # Check arguments
        if len(test_images) != len(true_characters):
            raise ValueError("Number of test images and true characters must match")
        
        with torch.no_grad():
            for i, (image, true_char) in enumerate(zip(test_images, true_characters)):
                # Preprocess image
                image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                image_tensor = image_tensor.to(self.device)
                
                # Direct OCR
                direct_result = recognize_single_character(image)
                
                # Deobfuscated OCR
                output = self.model(image_tensor)
                output_np = output.cpu().numpy()[0, 0]
                output_img = (output_np * 255).astype(np.uint8)
                _, output_bin = cv2.threshold(output_img, int(self.threshold * 255), 255, cv2.THRESH_BINARY)
                deobfuscated_result = recognize_single_character(output_bin)
                
                # Store results
                results['direct_ocr'].append({
                    'predicted': direct_result,
                    'true': true_char,
                    'correct': direct_result == true_char
                })
                
                results['deobfuscated_ocr'].append({
                    'predicted': deobfuscated_result,
                    'true': true_char,
                    'correct': deobfuscated_result == true_char
                })
        
        # Calculate metrics
        direct_correct = sum(1 for r in results['direct_ocr'] if r['correct'])
        deobfuscated_correct = sum(1 for r in results['deobfuscated_ocr'] if r['correct'])
        
        num_samples = len(true_characters)
        direct_accuracy = direct_correct / num_samples
        deobfuscated_accuracy = deobfuscated_correct / num_samples
        improvement = (deobfuscated_accuracy - direct_accuracy) / max(direct_accuracy, 1e-8)
        
        results['metrics'] = {
            'direct_accuracy': direct_accuracy,
            'deobfuscated_accuracy': deobfuscated_accuracy,
            'improvement': improvement,
            'num_samples': num_samples
        }
        
        # Save results
        if save_results:
            # Save metrics as JSON
            comparison_path = self.output_dir / "pipeline_comparison.json"
            with open(comparison_path, 'w') as f:
                json.dump(results['metrics'], f, indent=2)
            
            # Create detailed results table
            detailed_results = []
            for i in range(num_samples):
                detailed_results.append({
                    'true_char': results['direct_ocr'][i]['true'],
                    'direct_ocr': results['direct_ocr'][i]['predicted'],
                    'direct_correct': results['direct_ocr'][i]['correct'],
                    'deobfuscated_ocr': results['deobfuscated_ocr'][i]['predicted'],
                    'deobfuscated_correct': results['deobfuscated_ocr'][i]['correct'],
                })
            
            # Save as CSV
            df = pd.DataFrame(detailed_results)
            df.to_csv(self.output_dir / "detailed_comparison.csv", index=False)
        
        logger.info(f"Pipeline comparison metrics: {json.dumps(results['metrics'], indent=2)}")
        
        return results
    
    def create_confusion_matrix(
        self,
        save_results: bool = True
    ) -> np.ndarray:
        """
        Create confusion matrix for OCR results.
        
        Args:
            save_results: Whether to save the confusion matrix visualization
            
        Returns:
            Confusion matrix
        """
        # Evaluate and get OCR results
        results = self.evaluate(save_results=False, save_samples=False, ocr_evaluation=True)
        
        if not results['ocr_results'] or 'ocr_results' not in results:
            logger.warning("No OCR results available for creating confusion matrix")
            return None
        
        # Extract predictions and truths
        y_true = []
        y_pred_direct = []
        y_pred_deobfuscated = []
        
        for result in results['ocr_results']['detailed_results']:
            if result['true_char']:
                y_true.append(result['true_char'])
                y_pred_direct.append(result['input_ocr'])
                y_pred_deobfuscated.append(result['output_ocr'])
        
        if not y_true:
            logger.warning("No valid OCR results with true characters")
            return None
        
        # Get unique characters
        unique_chars = sorted(set(y_true + y_pred_direct + y_pred_deobfuscated))
        
        # Create confusion matrices
        cm_direct = confusion_matrix(y_true, y_pred_direct, labels=unique_chars)
        cm_deobfuscated = confusion_matrix(y_true, y_pred_deobfuscated, labels=unique_chars)
        
        # Calculate precision, recall, F1
        precision_direct, recall_direct, f1_direct, _ = precision_recall_fscore_support(
            y_true, y_pred_direct, labels=unique_chars, average=None
        )
        precision_deob, recall_deob, f1_deob, _ = precision_recall_fscore_support(
            y_true, y_pred_deobfuscated, labels=unique_chars, average=None
        )
        
        # Save results
        if save_results:
            # Plot confusion matrices
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            # Direct OCR confusion matrix
            im0 = axes[0].imshow(cm_direct, cmap='Blues')
            axes[0].set_title('Direct OCR Confusion Matrix')
            axes[0].set_xticks(np.arange(len(unique_chars)))
            axes[0].set_yticks(np.arange(len(unique_chars)))
            axes[0].set_xticklabels(unique_chars)
            axes[0].set_yticklabels(unique_chars)
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('True')
            plt.colorbar(im0, ax=axes[0])
            
            # Deobfuscated OCR confusion matrix
            im1 = axes[1].imshow(cm_deobfuscated, cmap='Blues')
            axes[1].set_title('Deobfuscated OCR Confusion Matrix')
            axes[1].set_xticks(np.arange(len(unique_chars)))
            axes[1].set_yticks(np.arange(len(unique_chars)))
            axes[1].set_xticklabels(unique_chars)
            axes[1].set_yticklabels(unique_chars)
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('True')
            plt.colorbar(im1, ax=axes[1])
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "confusion_matrices.png", dpi=200)
            plt.close()
            
            # Save metrics in a DataFrame
            metrics_df = pd.DataFrame({
                'character': unique_chars,
                'precision_direct': precision_direct,
                'recall_direct': recall_direct,
                'f1_direct': f1_direct,
                'precision_deobfuscated': precision_deob,
                'recall_deobfuscated': recall_deob,
                'f1_deobfuscated': f1_deob,
            })
            metrics_df.to_csv(self.output_dir / "character_metrics.csv", index=False)
        
        logger.info("Confusion matrices and metrics created")
        
        return {
            'direct': cm_direct,
            'deobfuscated': cm_deobfuscated,
            'characters': unique_chars
        }
