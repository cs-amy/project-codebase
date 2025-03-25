"""
Image processing utility functions for the OCR deobfuscation project.

This module contains functions for loading, saving, preprocessing, and augmenting
images throughout the pipeline, including resizing, normalization, thresholding,
denoising, and various transformations.
"""

import os
import numpy as np
import cv2
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image to a target size.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        target_size (Tuple[int, int]): Target size as (height, width).
        
    Returns:
        np.ndarray: Resized image.
    """
    if image.shape[:2] == target_size:
        return image
    
    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image to the range [0, 1].
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        
    Returns:
        np.ndarray: Normalized image.
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        if np.max(image) <= 1.0 and np.min(image) >= 0.0:
            return image
    
    image = image.astype(np.float32)
    
    if np.max(image) > 0:
        image = image / 255.0
    
    return image

def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert a normalized [0, 1] image back to [0, 255] range.
    
    Args:
        image (np.ndarray): Normalized input image as a numpy array.
        
    Returns:
        np.ndarray: Denormalized image.
    """
    if image.dtype == np.uint8:
        return image
    
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8)
    
    return image

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale if it has multiple channels.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        
    Returns:
        np.ndarray: Grayscale image.
    """
    if len(image.shape) == 2:
        return image
    
    if image.shape[2] == 1:
        return image[:, :, 0]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def apply_threshold(image: np.ndarray, threshold: int = 127, max_value: int = 255) -> np.ndarray:
    """
    Apply binary thresholding to an image.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        threshold (int, optional): Threshold value. Defaults to 127.
        max_value (int, optional): Maximum value to use with the threshold. Defaults to 255.
        
    Returns:
        np.ndarray: Thresholded image.
    """
    # Ensure image is grayscale
    if len(image.shape) > 2:
        image = convert_to_grayscale(image)
    
    # Ensure image is in [0, 255] range
    if image.dtype != np.uint8:
        image = denormalize_image(image)
    
    _, binary = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)
    return binary

def adaptive_threshold(image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """
    Apply adaptive thresholding to an image.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        block_size (int, optional): Size of the neighborhood area. Defaults to 11.
        c (int, optional): Constant subtracted from the mean. Defaults to 2.
        
    Returns:
        np.ndarray: Thresholded image.
    """
    # Ensure image is grayscale
    if len(image.shape) > 2:
        image = convert_to_grayscale(image)
    
    # Ensure image is in [0, 255] range
    if image.dtype != np.uint8:
        image = denormalize_image(image)
    
    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1
    
    adaptive = cv2.adaptiveThreshold(
        image, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        block_size, 
        c
    )
    
    return adaptive

def add_gaussian_noise(image: np.ndarray, mean: float = 0, std: float = 0.05) -> np.ndarray:
    """
    Add Gaussian noise to an image.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        mean (float, optional): Mean of the Gaussian noise. Defaults to 0.
        std (float, optional): Standard deviation of the Gaussian noise. Defaults to 0.05.
        
    Returns:
        np.ndarray: Noisy image.
    """
    # Convert to float if needed
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = normalize_image(image)
    
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    
    # Clip values to valid range
    noisy_image = np.clip(noisy_image, 0.0, 1.0)
    
    return noisy_image

def random_rotation(image: np.ndarray, max_angle: float = 10) -> np.ndarray:
    """
    Apply a random rotation to an image.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        max_angle (float, optional): Maximum rotation angle in degrees. Defaults to 10.
        
    Returns:
        np.ndarray: Rotated image.
    """
    angle = np.random.uniform(-max_angle, max_angle)
    
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply the rotation
    rotated = cv2.warpAffine(
        image, 
        rotation_matrix, 
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255 if image.dtype == np.uint8 else 1.0
    )
    
    return rotated

def center_crop(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Crop an image to the target size from the center.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        target_size (Tuple[int, int]): Target size as (height, width).
        
    Returns:
        np.ndarray: Cropped image.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Check if crop is possible
    if h < target_h or w < target_w:
        logger.warning(f"Cannot crop image of size {(h, w)} to target size {target_size}")
        return resize_image(image, target_size)
    
    # Calculate crop coordinates
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    
    # Perform crop
    cropped = image[start_h:start_h+target_h, start_w:start_w+target_w]
    
    return cropped

def pad_image(image: np.ndarray, target_size: Tuple[int, int], pad_value: int = 0) -> np.ndarray:
    """
    Pad an image to the target size.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        target_size (Tuple[int, int]): Target size as (height, width).
        pad_value (int, optional): Value to use for padding. Defaults to 0.
        
    Returns:
        np.ndarray: Padded image.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Check if padding is needed
    if h >= target_h and w >= target_w:
        return center_crop(image, target_size)
    
    # Convert pad_value to float if the image is float
    if image.dtype == np.float32 or image.dtype == np.float64:
        pad_value = float(pad_value) / 255.0 if pad_value > 1.0 else float(pad_value)
    
    # Create a blank canvas
    if len(image.shape) == 3:
        padded = np.ones((target_h, target_w, image.shape[2]), dtype=image.dtype) * pad_value
    else:
        padded = np.ones((target_h, target_w), dtype=image.dtype) * pad_value
    
    # Calculate paste coordinates
    start_h = max(0, (target_h - h) // 2)
    start_w = max(0, (target_w - w) // 2)
    
    # Calculate which part of the image to use
    img_start_h = max(0, -((target_h - h) // 2))
    img_start_w = max(0, -((target_w - w) // 2))
    img_end_h = min(h, img_start_h + (target_h - start_h))
    img_end_w = min(w, img_start_w + (target_w - start_w))
    
    # Paste the image onto the canvas
    paste_h = min(target_h - start_h, img_end_h - img_start_h)
    paste_w = min(target_w - start_w, img_end_w - img_start_w)
    
    if len(image.shape) == 3:
        padded[start_h:start_h+paste_h, start_w:start_w+paste_w, :] = \
            image[img_start_h:img_start_h+paste_h, img_start_w:img_start_w+paste_w, :]
    else:
        padded[start_h:start_h+paste_h, start_w:start_w+paste_w] = \
            image[img_start_h:img_start_h+paste_h, img_start_w:img_start_w+paste_w]
    
    return padded

def random_shift(image: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
    """
    Apply a random shift to an image.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        max_shift (float, optional): Maximum shift as a fraction of the image size. Defaults to 0.1.
        
    Returns:
        np.ndarray: Shifted image.
    """
    height, width = image.shape[:2]
    
    # Calculate shift amount
    shift_h = int(height * np.random.uniform(-max_shift, max_shift))
    shift_w = int(width * np.random.uniform(-max_shift, max_shift))
    
    # Create the transformation matrix
    M = np.float32([[1, 0, shift_w], [0, 1, shift_h]])
    
    # Apply the shift
    shifted = cv2.warpAffine(
        image, 
        M, 
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255 if image.dtype == np.uint8 else 1.0
    )
    
    return shifted

def random_zoom(image: np.ndarray, zoom_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """
    Apply a random zoom to an image.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        zoom_range (Tuple[float, float], optional): Range of zoom factors. Defaults to (0.9, 1.1).
        
    Returns:
        np.ndarray: Zoomed image.
    """
    height, width = image.shape[:2]
    
    # Get random zoom factor
    zoom = np.random.uniform(zoom_range[0], zoom_range[1])
    
    # Calculate new dimensions
    new_h, new_w = int(height * zoom), int(width * zoom)
    
    # Resize the image
    if zoom > 1.0:
        # Zoom in: resize and then crop
        zoomed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        zoomed = center_crop(zoomed, (height, width))
    else:
        # Zoom out: resize to smaller and then pad
        zoomed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        zoomed = pad_image(
            zoomed, 
            (height, width),
            pad_value=255 if image.dtype == np.uint8 else 1.0
        )
    
    return zoomed

def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust the brightness of an image.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        factor (float): Brightness adjustment factor. Values > 1 increase brightness.
        
    Returns:
        np.ndarray: Brightness-adjusted image.
    """
    # Convert to float for adjustment
    if image.dtype != np.float32 and image.dtype != np.float64:
        was_uint8 = True
        image = normalize_image(image)
    else:
        was_uint8 = False
    
    # Apply brightness adjustment
    adjusted = image * factor
    
    # Clip values to valid range
    adjusted = np.clip(adjusted, 0.0, 1.0)
    
    # Convert back to original type if needed
    if was_uint8:
        adjusted = denormalize_image(adjusted)
    
    return adjusted

def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust the contrast of an image.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        factor (float): Contrast adjustment factor. Values > 1 increase contrast.
        
    Returns:
        np.ndarray: Contrast-adjusted image.
    """
    # Convert to float for adjustment
    if image.dtype != np.float32 and image.dtype != np.float64:
        was_uint8 = True
        image = normalize_image(image)
    else:
        was_uint8 = False
    
    # Calculate mean
    mean = np.mean(image)
    
    # Apply contrast adjustment
    adjusted = mean + factor * (image - mean)
    
    # Clip values to valid range
    adjusted = np.clip(adjusted, 0.0, 1.0)
    
    # Convert back to original type if needed
    if was_uint8:
        adjusted = denormalize_image(adjusted)
    
    return adjusted

def denoise_image(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Apply denoising to an image.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        strength (int, optional): Strength of the denoising. Defaults to 10.
        
    Returns:
        np.ndarray: Denoised image.
    """
    # Ensure image is in correct format
    if image.dtype != np.uint8:
        image = denormalize_image(image)
    
    # Apply different denoising based on channels
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Color image
        denoised = cv2.fastNlMeansDenoisingColored(
            image, 
            None, 
            strength, 
            strength, 
            7, 
            21
        )
    else:
        # Grayscale image
        if len(image.shape) == 3:
            image = image[:, :, 0]
        denoised = cv2.fastNlMeansDenoising(
            image, 
            None, 
            strength, 
            7, 
            21
        )
        
        # Reshape if needed
        if len(image.shape) == 3:
            denoised = denoised.reshape(image.shape)
    
    return denoised

def load_image(image_path: str, grayscale: bool = True, normalize: bool = True) -> np.ndarray:
    """
    Load an image from a file path.
    
    Args:
        image_path (str): Path to the image file.
        grayscale (bool, optional): Whether to convert to grayscale. Defaults to True.
        normalize (bool, optional): Whether to normalize to [0, 1]. Defaults to True.
        
    Returns:
        np.ndarray: Loaded image.
        
    Raises:
        FileNotFoundError: If the image file is not found.
        ValueError: If the image cannot be loaded.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Normalize if requested
        if normalize:
            image = normalize_image(image)
        
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise

def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save an image to a file.
    
    Args:
        image (np.ndarray): Image to save.
        output_path (str): Path where to save the image.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Ensure image is in the right format for saving
    if image.dtype != np.uint8:
        image = denormalize_image(image)
    
    # Save the image
    cv2.imwrite(output_path, image)

def apply_morphology(image: np.ndarray, operation: str = 'dilate', kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Apply morphological operations to an image.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        operation (str, optional): Operation type ('erode', 'dilate', 'open', 'close'). Defaults to 'dilate'.
        kernel_size (int, optional): Size of the kernel. Defaults to 3.
        iterations (int, optional): Number of iterations. Defaults to 1.
        
    Returns:
        np.ndarray: Processed image.
    """
    # Ensure image is in the right format
    if image.dtype != np.uint8:
        image = denormalize_image(image)
    
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply operation
    if operation == 'erode':
        processed = cv2.erode(image, kernel, iterations=iterations)
    elif operation == 'dilate':
        processed = cv2.dilate(image, kernel, iterations=iterations)
    elif operation == 'open':
        processed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'close':
        processed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        logger.warning(f"Unknown morphological operation: {operation}. Using dilate.")
        processed = cv2.dilate(image, kernel, iterations=iterations)
    
    return processed

def apply_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply Gaussian blur to an image.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        kernel_size (int, optional): Size of the blur kernel. Defaults to 3.
        
    Returns:
        np.ndarray: Blurred image.
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Apply blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    return blurred

def prepare_for_model(image: np.ndarray, target_size: Tuple[int, int] = (64, 64), normalize: bool = True) -> np.ndarray:
    """
    Prepare an image for input to a model.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        target_size (Tuple[int, int], optional): Target size as (height, width). Defaults to (64, 64).
        normalize (bool, optional): Whether to normalize pixel values. Defaults to True.
        
    Returns:
        np.ndarray: Prepared image.
    """
    # Convert to grayscale
    gray = convert_to_grayscale(image)
    
    # Resize
    resized = resize_image(gray, target_size)
    
    # Normalize
    if normalize:
        normalized = normalize_image(resized)
    else:
        normalized = resized
    
    # Add channel dimension if needed
    if len(normalized.shape) == 2:
        normalized = normalized.reshape((*normalized.shape, 1))
    
    return normalized

def augment_image(image: np.ndarray, params: dict) -> np.ndarray:
    """
    Apply a series of augmentations to an image based on parameters.
    
    Args:
        image (np.ndarray): Input image as a numpy array.
        params (dict): Augmentation parameters.
            - rotation_range (float): Maximum rotation angle in degrees.
            - width_shift_range (float): Maximum horizontal shift as fraction of width.
            - height_shift_range (float): Maximum vertical shift as fraction of height.
            - zoom_range (Tuple[float, float]): Range of zoom factors.
            - brightness_range (Tuple[float, float]): Range of brightness factors.
            - contrast_range (Tuple[float, float]): Range of contrast factors.
            - noise_factor (float): Standard deviation of Gaussian noise.
        
    Returns:
        np.ndarray: Augmented image.
    """
    augmented = image.copy()
    
    # Apply rotation
    if 'rotation_range' in params and np.random.random() < 0.5:
        augmented = random_rotation(augmented, params['rotation_range'])
    
    # Apply shift
    if ('width_shift_range' in params or 'height_shift_range' in params) and np.random.random() < 0.5:
        max_shift = max(
            params.get('width_shift_range', 0),
            params.get('height_shift_range', 0)
        )
        augmented = random_shift(augmented, max_shift)
    
    # Apply zoom
    if 'zoom_range' in params and np.random.random() < 0.5:
        zoom_range = params['zoom_range']
        if isinstance(zoom_range, (int, float)):
            zoom_range = (1 - zoom_range, 1 + zoom_range)
        augmented = random_zoom(augmented, zoom_range)
    
    # Apply brightness adjustment
    if 'brightness_range' in params and np.random.random() < 0.5:
        factor = np.random.uniform(*params['brightness_range'])
        augmented = adjust_brightness(augmented, factor)
    
    # Apply contrast adjustment
    if 'contrast_range' in params and np.random.random() < 0.5:
        factor = np.random.uniform(*params['contrast_range'])
        augmented = adjust_contrast(augmented, factor)
    
    # Apply noise
    if 'noise_factor' in params and np.random.random() < 0.5:
        augmented = add_gaussian_noise(augmented, std=params['noise_factor'])
    
    return augmented

if __name__ == "__main__":
    # Simple demo/test code
    logging.basicConfig(level=logging.INFO)
    
    # Create a test image (white with a black square)
    test_image = np.ones((100, 100), dtype=np.uint8) * 255
    test_image[25:75, 25:75] = 0
    
    # Test various functions
    logger.info("Testing image processing functions...")
    
    # Resize
    resized = resize_image(test_image, (64, 64))
    logger.info(f"Resized shape: {resized.shape}")
    
    # Normalize and denormalize
    normalized = normalize_image(test_image)
    logger.info(f"Normalized range: [{np.min(normalized)}, {np.max(normalized)}]")
    denormalized = denormalize_image(normalized)
    logger.info(f"Denormalized range: [{np.min(denormalized)}, {np.max(denormalized)}]")
    
    # Thresholding
    binary = apply_threshold(test_image, 128)
    logger.info(f"Binary unique values: {np.unique(binary)}")
    
    # Augmentations
    rotated = random_rotation(test_image, 45)
    logger.info(f"Rotated shape: {rotated.shape}")
    
    shifted = random_shift(test_image, 0.2)
    logger.info(f"Shifted shape: {shifted.shape}")
    
    zoomed = random_zoom(test_image, (0.8, 1.2))
    logger.info(f"Zoomed shape: {zoomed.shape}")
    
    # Create a temp directory for saving test images
    os.makedirs("temp", exist_ok=True)
    
    # Save test images
    cv2.imwrite("temp/original.png", test_image)
    cv2.imwrite("temp/resized.png", resized)
    cv2.imwrite("temp/binary.png", binary)
    cv2.imwrite("temp/rotated.png", rotated)
    cv2.imwrite("temp/shifted.png", shifted)
    cv2.imwrite("temp/zoomed.png", zoomed)
    
    logger.info("Saved test images to 'temp/' directory")
    
    # Test augmentation
    aug_params = {
        'rotation_range': 15,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'zoom_range': (0.9, 1.1),
        'brightness_range': (0.9, 1.1),
        'contrast_range': (0.9, 1.1),
        'noise_factor': 0.05
    }
    
    augmented = augment_image(test_image, aug_params)
    cv2.imwrite("temp/augmented.png", augmented)
    logger.info("Saved augmented image to 'temp/augmented.png'") 