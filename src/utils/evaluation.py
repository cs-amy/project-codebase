"""
Evaluation metrics and utilities for the OCR deobfuscation project.

This module provides functions to compute various evaluation metrics for assessing
the performance of deobfuscation models, including pixel-level metrics (accuracy, 
Dice coefficient, IoU), structural metrics (SSIM), and OCR-specific metrics.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix
import cv2
from skimage.metrics import structural_similarity as ssim
import Levenshtein

logger = logging.getLogger(__name__)

def pixel_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculate pixel-wise accuracy between prediction and ground truth.
    
    Args:
        y_true (np.ndarray): Ground truth binary image.
        y_pred (np.ndarray): Predicted image (can be float [0,1] or binary).
        threshold (float, optional): Threshold to binarize predicted image. Defaults to 0.5.
        
    Returns:
        float: Pixel accuracy (0.0-1.0).
    """
    # Ensure shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    # Binarize predictions
    if y_pred.dtype == np.float32 or y_pred.dtype == np.float64:
        y_pred_binary = (y_pred > threshold).astype(np.uint8)
    else:
        y_pred_binary = y_pred
    
    # Ensure y_true is binary
    if y_true.dtype == np.float32 or y_true.dtype == np.float64:
        y_true_binary = (y_true > threshold).astype(np.uint8)
    else:
        y_true_binary = y_true
    
    # Calculate accuracy
    correct = np.sum(y_true_binary == y_pred_binary)
    total = y_true.size
    
    return float(correct) / total

def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, 
                     smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient (F1 score) between prediction and ground truth.
    
    Args:
        y_true (np.ndarray): Ground truth binary image.
        y_pred (np.ndarray): Predicted image (can be float [0,1] or binary).
        threshold (float, optional): Threshold to binarize predicted image. Defaults to 0.5.
        smooth (float, optional): Smoothing factor to avoid division by zero. Defaults to 1e-6.
        
    Returns:
        float: Dice coefficient (0.0-1.0).
    """
    # Ensure shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    # Binarize predictions
    if y_pred.dtype == np.float32 or y_pred.dtype == np.float64:
        y_pred_binary = (y_pred > threshold).astype(np.uint8)
    else:
        y_pred_binary = y_pred
    
    # Ensure y_true is binary
    if y_true.dtype == np.float32 or y_true.dtype == np.float64:
        y_true_binary = (y_true > threshold).astype(np.uint8)
    else:
        y_true_binary = y_true
    
    # Calculate intersection and union
    intersection = np.sum(y_true_binary * y_pred_binary)
    true_sum = np.sum(y_true_binary)
    pred_sum = np.sum(y_pred_binary)
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (true_sum + pred_sum + smooth)
    
    return float(dice)

def iou_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, 
              smooth: float = 1e-6) -> float:
    """
    Calculate Intersection over Union (IoU) score between prediction and ground truth.
    
    Args:
        y_true (np.ndarray): Ground truth binary image.
        y_pred (np.ndarray): Predicted image (can be float [0,1] or binary).
        threshold (float, optional): Threshold to binarize predicted image. Defaults to 0.5.
        smooth (float, optional): Smoothing factor to avoid division by zero. Defaults to 1e-6.
        
    Returns:
        float: IoU score (0.0-1.0).
    """
    # Ensure shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    # Binarize predictions
    if y_pred.dtype == np.float32 or y_pred.dtype == np.float64:
        y_pred_binary = (y_pred > threshold).astype(np.uint8)
    else:
        y_pred_binary = y_pred
    
    # Ensure y_true is binary
    if y_true.dtype == np.float32 or y_true.dtype == np.float64:
        y_true_binary = (y_true > threshold).astype(np.uint8)
    else:
        y_true_binary = y_true
    
    # Calculate intersection and union
    intersection = np.sum(y_true_binary * y_pred_binary)
    union = np.sum(y_true_binary) + np.sum(y_pred_binary) - intersection
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return float(iou)

def structural_similarity_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between prediction and ground truth.
    
    Args:
        y_true (np.ndarray): Ground truth image.
        y_pred (np.ndarray): Predicted image.
        
    Returns:
        float: SSIM score (-1.0 to 1.0, higher is better).
    """
    # Ensure shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    # Convert to grayscale if needed
    if len(y_true.shape) > 2 and y_true.shape[2] > 1:
        y_true_gray = cv2.cvtColor(y_true, cv2.COLOR_BGR2GRAY)
    else:
        y_true_gray = y_true.squeeze()
    
    if len(y_pred.shape) > 2 and y_pred.shape[2] > 1:
        y_pred_gray = cv2.cvtColor(y_pred, cv2.COLOR_BGR2GRAY)
    else:
        y_pred_gray = y_pred.squeeze()
    
    # Ensure values are in the correct range for SSIM
    if y_true_gray.dtype != np.uint8:
        y_true_gray = (y_true_gray * 255).astype(np.uint8)
    
    if y_pred_gray.dtype != np.uint8:
        y_pred_gray = (y_pred_gray * 255).astype(np.uint8)
    
    # Calculate SSIM
    score, _ = ssim(y_true_gray, y_pred_gray, full=True)
    
    return float(score)

def character_recognition_accuracy(true_chars: List[str], pred_chars: List[str]) -> float:
    """
    Calculate character-level recognition accuracy.
    
    Args:
        true_chars (List[str]): List of true characters.
        pred_chars (List[str]): List of predicted characters.
        
    Returns:
        float: Character recognition accuracy (0.0-1.0).
    """
    if len(true_chars) != len(pred_chars):
        logger.warning(f"Length mismatch: true_chars {len(true_chars)} vs pred_chars {len(pred_chars)}")
        # Truncate to shorter length
        min_len = min(len(true_chars), len(pred_chars))
        true_chars = true_chars[:min_len]
        pred_chars = pred_chars[:min_len]
    
    correct = sum(t == p for t, p in zip(true_chars, pred_chars))
    total = len(true_chars)
    
    return float(correct) / total if total > 0 else 0.0

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.
    
    Args:
        s1 (str): First string.
        s2 (str): Second string.
        
    Returns:
        int: Levenshtein distance.
    """
    return Levenshtein.distance(s1, s2)

def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calculate the Levenshtein similarity between two strings.
    
    Args:
        s1 (str): First string.
        s2 (str): Second string.
        
    Returns:
        float: Levenshtein similarity (0.0-1.0).
    """
    if not s1 and not s2:
        return 1.0
    
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    
    distance = levenshtein_distance(s1, s2)
    similarity = 1.0 - (distance / max_len)
    
    return similarity

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate all pixel-level metrics between prediction and ground truth.
    
    Args:
        y_true (np.ndarray): Ground truth image.
        y_pred (np.ndarray): Predicted image.
        threshold (float, optional): Threshold to binarize predicted image. Defaults to 0.5.
        
    Returns:
        Dict[str, float]: Dictionary containing all metrics.
    """
    metrics = {}
    
    try:
        metrics['pixel_accuracy'] = pixel_accuracy(y_true, y_pred, threshold)
    except Exception as e:
        logger.error(f"Error calculating pixel accuracy: {e}")
        metrics['pixel_accuracy'] = 0.0
    
    try:
        metrics['dice_coefficient'] = dice_coefficient(y_true, y_pred, threshold)
    except Exception as e:
        logger.error(f"Error calculating Dice coefficient: {e}")
        metrics['dice_coefficient'] = 0.0
    
    try:
        metrics['iou_score'] = iou_score(y_true, y_pred, threshold)
    except Exception as e:
        logger.error(f"Error calculating IoU score: {e}")
        metrics['iou_score'] = 0.0
    
    try:
        metrics['ssim'] = structural_similarity_index(y_true, y_pred)
    except Exception as e:
        logger.error(f"Error calculating SSIM: {e}")
        metrics['ssim'] = 0.0
    
    return metrics

def character_confusion_matrix(true_chars: List[str], pred_chars: List[str], 
                              classes: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Calculate confusion matrix for character recognition.
    
    Args:
        true_chars (List[str]): List of true characters.
        pred_chars (List[str]): List of predicted characters.
        classes (Optional[List[str]], optional): List of class labels. 
                                             If None, derived from data. Defaults to None.
        
    Returns:
        Tuple[np.ndarray, List[str]]: Confusion matrix and list of class labels.
    """
    if len(true_chars) != len(pred_chars):
        logger.warning(f"Length mismatch: true_chars {len(true_chars)} vs pred_chars {len(pred_chars)}")
        # Truncate to shorter length
        min_len = min(len(true_chars), len(pred_chars))
        true_chars = true_chars[:min_len]
        pred_chars = pred_chars[:min_len]
    
    # Determine class labels if not provided
    if classes is None:
        classes = sorted(list(set(true_chars) | set(pred_chars)))
    
    # Create label encoder
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    
    # Convert characters to indices
    true_indices = [label_to_idx.get(char, len(classes)) for char in true_chars]
    pred_indices = [label_to_idx.get(char, len(classes)) for char in pred_chars]
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_indices, pred_indices, labels=list(range(len(classes))))
    
    return cm, classes

def ocr_word_accuracy(true_words: List[str], pred_words: List[str], 
                     case_sensitive: bool = False) -> Dict[str, float]:
    """
    Calculate word-level OCR accuracy metrics.
    
    Args:
        true_words (List[str]): List of true words.
        pred_words (List[str]): List of predicted words.
        case_sensitive (bool, optional): Whether comparison is case-sensitive. Defaults to False.
        
    Returns:
        Dict[str, float]: Dictionary containing word-level metrics.
    """
    if len(true_words) != len(pred_words):
        logger.warning(f"Length mismatch: true_words {len(true_words)} vs pred_words {len(pred_words)}")
        # Truncate to shorter length
        min_len = min(len(true_words), len(pred_words))
        true_words = true_words[:min_len]
        pred_words = pred_words[:min_len]
    
    # Copy lists to avoid modifying originals
    true_words_copy = true_words.copy()
    pred_words_copy = pred_words.copy()
    
    # Convert to lowercase if not case-sensitive
    if not case_sensitive:
        true_words_copy = [w.lower() for w in true_words_copy]
        pred_words_copy = [w.lower() for w in pred_words_copy]
    
    # Calculate exact match accuracy
    exact_matches = sum(t == p for t, p in zip(true_words_copy, pred_words_copy))
    total_words = len(true_words_copy)
    
    # Calculate Levenshtein similarities
    similarities = [levenshtein_similarity(t, p) for t, p in zip(true_words_copy, pred_words_copy)]
    
    return {
        'exact_match_accuracy': float(exact_matches) / total_words if total_words > 0 else 0.0,
        'mean_levenshtein_similarity': float(sum(similarities)) / total_words if total_words > 0 else 0.0,
        'min_levenshtein_similarity': float(min(similarities)) if similarities else 0.0,
        'max_levenshtein_similarity': float(max(similarities)) if similarities else 0.0
    }

def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics from multiple samples.
    
    Args:
        metrics_list (List[Dict[str, float]]): List of metric dictionaries from multiple samples.
        
    Returns:
        Dict[str, float]: Aggregated metrics with mean, min, max, and std.
    """
    if not metrics_list:
        return {}
    
    # Get all unique metric names
    all_metrics = set()
    for metrics in metrics_list:
        all_metrics.update(metrics.keys())
    
    # Initialize aggregated metrics
    aggregated = {}
    
    # Aggregate each metric
    for metric_name in all_metrics:
        # Collect all values for this metric
        values = [
            metrics[metric_name] for metrics in metrics_list 
            if metric_name in metrics and metrics[metric_name] is not None
        ]
        
        if values:
            aggregated[f"{metric_name}_mean"] = float(np.mean(values))
            aggregated[f"{metric_name}_min"] = float(np.min(values))
            aggregated[f"{metric_name}_max"] = float(np.max(values))
            aggregated[f"{metric_name}_std"] = float(np.std(values))
    
    return aggregated

if __name__ == "__main__":
    # Simple demo/test code
    logging.basicConfig(level=logging.INFO)
    
    # Create test images
    true_img = np.zeros((64, 64), dtype=np.uint8)
    true_img[16:48, 16:48] = 255  # White square in the middle
    
    pred_img1 = np.zeros((64, 64), dtype=np.uint8)
    pred_img1[20:52, 20:52] = 255  # Slightly shifted square
    
    pred_img2 = np.zeros((64, 64), dtype=np.uint8)
    pred_img2[16:48, 16:48] = 255  # Exactly matching square
    
    # Test pixel-level metrics
    metrics1 = calculate_all_metrics(true_img, pred_img1)
    metrics2 = calculate_all_metrics(true_img, pred_img2)
    
    logger.info("Metrics for shifted prediction:")
    for name, value in metrics1.items():
        logger.info(f"  {name}: {value:.4f}")
    
    logger.info("Metrics for exact prediction:")
    for name, value in metrics2.items():
        logger.info(f"  {name}: {value:.4f}")
    
    # Test character-level metrics
    true_chars = ['a', 'b', 'c', 'd', 'e']
    pred_chars1 = ['a', 'b', 'x', 'd', 'f']
    pred_chars2 = ['a', 'b', 'c', 'd', 'e']
    
    char_acc1 = character_recognition_accuracy(true_chars, pred_chars1)
    char_acc2 = character_recognition_accuracy(true_chars, pred_chars2)
    
    logger.info(f"Character accuracy (with errors): {char_acc1:.4f}")
    logger.info(f"Character accuracy (perfect): {char_acc2:.4f}")
    
    # Test confusion matrix
    cm, classes = character_confusion_matrix(true_chars, pred_chars1)
    logger.info(f"Confusion matrix classes: {classes}")
    logger.info(f"Confusion matrix shape: {cm.shape}")
    
    # Test word-level metrics
    true_words = ['hello', 'world', 'this', 'is', 'a', 'test']
    pred_words = ['hello', 'world', 'thas', 'is', 'x', 'test']
    
    word_metrics = ocr_word_accuracy(true_words, pred_words)
    logger.info("Word-level metrics:")
    for name, value in word_metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    
    # Test metric aggregation
    all_metrics = [metrics1, metrics2]
    agg_metrics = aggregate_metrics(all_metrics)
    
    logger.info("Aggregated metrics:")
    for name, value in agg_metrics.items():
        logger.info(f"  {name}: {value:.4f}") 