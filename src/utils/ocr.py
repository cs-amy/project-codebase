"""
Utility functions for working with Tesseract OCR.
"""

import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import os

from src.utils.image_processing import normalize_image, denormalize_image


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image for optimal OCR performance.
    
    Args:
        image: Input image as a numpy array
        
    Returns:
        Preprocessed image ready for OCR
    """
    # Ensure image is grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize if needed
    if gray.dtype != np.uint8:
        gray = denormalize_image(gray)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Optional: noise removal
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return opening


def recognize_text(image: np.ndarray, lang: str = 'eng', config: str = '--psm 10') -> str:
    """
    Recognize text in an image using Tesseract OCR.
    
    Args:
        image: Input image as a numpy array
        lang: Language for OCR (default: 'eng')
        config: Tesseract configuration string
            - PSM modes:
              - 0: Orientation and script detection only
              - 1: Automatic page segmentation with OSD
              - 3: Fully automatic page segmentation, but no OSD (default)
              - 4: Assume a single column of text of variable sizes
              - 6: Assume a single uniform block of text
              - 7: Treat the image as a single text line
              - 8: Treat the image as a single word
              - 9: Treat the image as a single word in a circle
              - 10: Treat the image as a single character
              - 11: Sparse text
              - 12: Sparse text with OSD
              - 13: Raw line
        
    Returns:
        Recognized text as a string
    """
    # Ensure image is in the right format (uint8)
    if image.dtype != np.uint8:
        image = denormalize_image(image)
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image)
    
    # Perform OCR
    text = pytesseract.image_to_string(pil_image, lang=lang, config=config).strip()
    
    return text


def recognize_single_character(image: np.ndarray) -> str:
    """
    Recognize a single character in an image.
    
    Args:
        image: Input image as a numpy array
        
    Returns:
        Recognized character as a string (empty string if no character detected)
    """
    # Preprocess image
    processed_image = preprocess_for_ocr(image)
    
    # Use tesseract with PSM mode 10 (single character)
    config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    
    # Perform OCR
    char = recognize_text(processed_image, config=config)
    
    # Return only the first character if something was recognized
    return char[0] if char else ""


def recognize_word(image: np.ndarray) -> str:
    """
    Recognize a single word in an image.
    
    Args:
        image: Input image as a numpy array
        
    Returns:
        Recognized word as a string
    """
    # Preprocess image
    processed_image = preprocess_for_ocr(image)
    
    # Use tesseract with PSM mode 8 (single word)
    config = '--psm 8'
    
    # Perform OCR
    word = recognize_text(processed_image, config=config)
    
    return word


def compare_ocr_results(original_text: str, recognized_text: str) -> Dict[str, float]:
    """
    Compare original text with OCR-recognized text and compute accuracy metrics.
    
    Args:
        original_text: Original text
        recognized_text: OCR-recognized text
        
    Returns:
        Dictionary of accuracy metrics
    """
    # Calculate character-level accuracy
    min_len = min(len(original_text), len(recognized_text))
    matching_chars = sum(a == b for a, b in zip(original_text[:min_len], recognized_text[:min_len]))
    
    # Character accuracy
    char_accuracy = matching_chars / len(original_text) if len(original_text) > 0 else 0.0
    
    # Word-level accuracy (exact match)
    word_accuracy = 1.0 if original_text == recognized_text else 0.0
    
    # Calculate edit distance (Levenshtein distance)
    edit_distance = levenshtein_distance(original_text, recognized_text)
    
    # Normalized edit distance (0 to 1, where 1 is perfect match)
    if len(original_text) > 0:
        normalized_distance = 1.0 - (edit_distance / max(len(original_text), len(recognized_text)))
    else:
        normalized_distance = 0.0
        
    return {
        'character_accuracy': char_accuracy,
        'word_accuracy': word_accuracy,
        'edit_distance': edit_distance,
        'normalized_distance': normalized_distance
    }


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance as an integer
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def get_tesseract_version() -> str:
    """
    Get the installed Tesseract version.
    
    Returns:
        Tesseract version as a string
    """
    try:
        version = pytesseract.get_tesseract_version()
        return str(version)
    except Exception as e:
        return f"Error getting Tesseract version: {str(e)}" 