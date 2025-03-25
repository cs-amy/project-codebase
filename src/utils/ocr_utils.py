#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OCR utility functions for text recognition.

This module provides functions to interact with Tesseract OCR engine,
process images for optimal OCR performance, and analyze OCR results.
"""

import os
import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Union, Optional
import difflib
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

def check_tesseract_installed() -> bool:
    """
    Check if Tesseract OCR is installed and accessible.
    
    Returns:
        bool: True if Tesseract is installed, False otherwise.
    """
    try:
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract OCR is not installed or not in PATH")
        return False
    except Exception as e:
        logger.error(f"Error checking Tesseract installation: {e}")
        return False

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image to improve OCR accuracy.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        
    Returns:
        np.ndarray: Preprocessed image.
    """
    # Convert to grayscale if image has multiple channels
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    return denoised

def perform_ocr(image: Union[np.ndarray, str, Image.Image], 
                lang: str = 'eng',
                config: str = '--psm 10 --oem 3',
                preprocess: bool = True) -> str:
    """
    Perform OCR on an image to extract text.
    
    Args:
        image (Union[np.ndarray, str, Image.Image]): Input image as numpy array, file path, or PIL Image.
        lang (str, optional): Language(s) to use for OCR. Defaults to 'eng'.
        config (str, optional): Configuration string for Tesseract. Defaults to '--psm 10 --oem 3'.
                               PSM 10 is for single character recognition.
        preprocess (bool, optional): Whether to preprocess the image. Defaults to True.
        
    Returns:
        str: Recognized text.
    """
    if not check_tesseract_installed():
        logger.error("Cannot perform OCR: Tesseract not installed")
        return ""
    
    # Handle different input types
    if isinstance(image, str):
        # It's a file path
        if not os.path.exists(image):
            logger.error(f"Image file not found: {image}")
            return ""
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):
        # It's a PIL Image
        image = np.array(image)
    
    # Preprocess the image if requested
    if preprocess:
        image = preprocess_for_ocr(image)
    
    # Perform OCR
    try:
        text = pytesseract.image_to_string(image, lang=lang, config=config).strip()
        return text
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""

def perform_ocr_with_confidence(image: Union[np.ndarray, str, Image.Image],
                                lang: str = 'eng',
                                config: str = '--psm 10 --oem 3') -> Dict:
    """
    Perform OCR and return the recognized text with confidence scores.
    
    Args:
        image (Union[np.ndarray, str, Image.Image]): Input image as numpy array, file path, or PIL Image.
        lang (str, optional): Language(s) to use for OCR. Defaults to 'eng'.
        config (str, optional): Configuration string for Tesseract. Defaults to '--psm 10 --oem 3'.
        
    Returns:
        Dict: Dictionary containing recognized text and confidence score.
              {'text': str, 'confidence': float}
    """
    if not check_tesseract_installed():
        logger.error("Cannot perform OCR: Tesseract not installed")
        return {"text": "", "confidence": 0.0}
    
    # Handle different input types
    if isinstance(image, str):
        # It's a file path
        if not os.path.exists(image):
            logger.error(f"Image file not found: {image}")
            return {"text": "", "confidence": 0.0}
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):
        # It's a PIL Image
        image = np.array(image)
    
    # Preprocess the image
    preprocessed = preprocess_for_ocr(image)
    
    # Perform OCR with detailed output
    try:
        data = pytesseract.image_to_data(preprocessed, lang=lang, config=config, output_type=pytesseract.Output.DICT)
        
        # Extract text and confidence from the first result
        if len(data['text']) > 0 and len(data['conf']) > 0:
            # Filter out empty results
            valid_indices = [i for i, t in enumerate(data['text']) if t.strip()]
            
            if valid_indices:
                text = data['text'][valid_indices[0]].strip()
                confidence = float(data['conf'][valid_indices[0]])
                return {"text": text, "confidence": confidence}
        
        return {"text": "", "confidence": 0.0}
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return {"text": "", "confidence": 0.0}

def calculate_ocr_accuracy(ground_truth: str, predicted: str) -> float:
    """
    Calculate accuracy between ground truth and OCR-predicted text.
    
    Args:
        ground_truth (str): The correct text.
        predicted (str): The OCR-predicted text.
        
    Returns:
        float: Accuracy score between 0.0 and 1.0.
    """
    if not ground_truth and not predicted:
        return 1.0
    elif not ground_truth or not predicted:
        return 0.0
    
    # Use difflib to calculate similarity
    matcher = difflib.SequenceMatcher(None, ground_truth, predicted)
    return matcher.ratio()

def create_confusion_matrix(ground_truths: List[str], 
                           predictions: List[str],
                           chars_to_include: Optional[str] = None) -> np.ndarray:
    """
    Create a confusion matrix for OCR results.
    
    Args:
        ground_truths (List[str]): List of ground truth characters.
        predictions (List[str]): List of predicted characters.
        chars_to_include (Optional[str], optional): String of characters to include in the matrix.
                                                  If None, uses all unique characters. Defaults to None.
        
    Returns:
        np.ndarray: Confusion matrix where rows are ground truths and columns are predictions.
    """
    # Filter out empty strings
    pairs = [(gt, pred) for gt, pred in zip(ground_truths, predictions) 
             if gt.strip() and len(gt) == 1]
    
    if not pairs:
        logger.warning("No valid character pairs for confusion matrix")
        return np.array([])
    
    # Extract unique characters
    if chars_to_include:
        unique_chars = sorted(set(chars_to_include))
    else:
        all_chars = set([p[0] for p in pairs]) | set([p[1] for p in pairs if p[1].strip()])
        unique_chars = sorted(all_chars)
    
    # Create character to index mapping
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    
    # Initialize confusion matrix
    n_chars = len(unique_chars)
    confusion_matrix = np.zeros((n_chars, n_chars), dtype=int)
    
    # Fill confusion matrix
    for gt, pred in pairs:
        if pred.strip():
            if pred not in char_to_idx:
                # Skip predictions not in our character set
                continue
            gt_idx = char_to_idx[gt]
            pred_idx = char_to_idx[pred]
            confusion_matrix[gt_idx, pred_idx] += 1
    
    return confusion_matrix, unique_chars

def ocr_pipeline_comparison(original_images: List[np.ndarray],
                           processed_images: List[np.ndarray],
                           ground_truths: List[str]) -> Dict:
    """
    Compare OCR performance on original vs processed images.
    
    Args:
        original_images (List[np.ndarray]): List of original images.
        processed_images (List[np.ndarray]): List of processed (deobfuscated) images.
        ground_truths (List[str]): List of ground truth texts.
        
    Returns:
        Dict: Dictionary with comparison metrics.
    """
    if not check_tesseract_installed():
        return {
            "error": "Tesseract not installed",
            "original_accuracy": 0.0,
            "processed_accuracy": 0.0,
            "improvement": 0.0
        }
    
    original_predictions = []
    processed_predictions = []
    original_accuracies = []
    processed_accuracies = []
    
    for orig_img, proc_img, truth in zip(original_images, processed_images, ground_truths):
        # OCR on original image
        orig_text = perform_ocr(orig_img)
        original_predictions.append(orig_text)
        orig_acc = calculate_ocr_accuracy(truth, orig_text)
        original_accuracies.append(orig_acc)
        
        # OCR on processed image
        proc_text = perform_ocr(proc_img)
        processed_predictions.append(proc_text)
        proc_acc = calculate_ocr_accuracy(truth, proc_text)
        processed_accuracies.append(proc_acc)
    
    # Calculate metrics
    mean_original_acc = np.mean(original_accuracies) if original_accuracies else 0.0
    mean_processed_acc = np.mean(processed_accuracies) if processed_accuracies else 0.0
    improvement = mean_processed_acc - mean_original_acc
    
    return {
        "original_accuracy": mean_original_acc,
        "processed_accuracy": mean_processed_acc,
        "improvement": improvement,
        "original_predictions": original_predictions,
        "processed_predictions": processed_predictions,
        "sample_count": len(ground_truths)
    }

if __name__ == "__main__":
    # Simple demo/test code
    logging.basicConfig(level=logging.INFO)
    
    if check_tesseract_installed():
        logger.info("Tesseract OCR is installed")
        # Generate a simple image with text for testing
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a white image
        img = Image.new('RGB', (100, 100), color='white')
        d = ImageDraw.Draw(img)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("Arial", 36)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw the letter 'A' in black
        d.text((35, 25), "A", fill='black', font=font)
        
        # Convert to numpy array for OCR
        img_array = np.array(img)
        
        # Perform OCR
        text = perform_ocr(img_array)
        logger.info(f"OCR result: '{text}'")
        
        # With confidence
        result = perform_ocr_with_confidence(img_array)
        logger.info(f"OCR result with confidence: {result}")
    else:
        logger.error("Tesseract OCR is not installed") 