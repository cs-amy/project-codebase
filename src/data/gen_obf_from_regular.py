"""
Script for generating obfuscated (blurred, rotated, etc.) character images from regular character images 
in the data/characters/regular directory.
This script reads the regular character images and creates corresponding obfuscated (blurred, rotated, etc.)
versions using the character mapping defined in the project.
"""

import os
import sys
import random
import argparse
import logging
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.image_processing import load_image, save_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate obfuscated character images from regular ones")
    
    parser.add_argument("--regular_dir", type=str, default="data/characters/regular",
                       help="Path to directory containing regular character images")
    parser.add_argument("--output_dir", type=str, default="data/characters/obfuscated",
                       help="Path to output directory for obfuscated images")
    parser.add_argument("--char_mapping_path", type=str, default="configs/character_mapping.yaml",
                       help="Path to character mapping YAML file")
    parser.add_argument("--num_samples_per_char", type=int, default=None,
                       help="Number of obfuscated samples to generate per character (default: same as regular)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    return parser.parse_args()


def load_character_mapping(mapping_path):
    """Load character mapping from YAML file."""
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = yaml.safe_load(f)
        
        # Filter out any comments or special sections
        if mapping:
            mapping = {k: v for k, v in mapping.items() if isinstance(k, str) and isinstance(v, list)}
        return mapping
    else:
        logger.warning(f"Character mapping file not found at {mapping_path}")
        return None


def apply_random_transformation(image):
    """
    Apply random transformations to an image to make it look different
    while preserving its general structure.
    
    Args:
        image: Input image
        
    Returns:
        Transformed image
    """
    # Make a copy to avoid modifying the original
    transformed = image.copy()
    
    # Random rotation (small angle)
    if random.random() > 0.5:
        angle = random.uniform(-10, 10)
        h, w = transformed.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        transformed = cv2.warpAffine(transformed, rotation_matrix, (w, h), 
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    # Random noise
    if random.random() > 0.7:
        noise = np.random.normal(0, 10, transformed.shape).astype(np.uint8)
        transformed = cv2.add(transformed, noise)
    
    # Random blur
    if random.random() > 0.7:
        kernel_size = random.choice([3, 5])
        transformed = cv2.GaussianBlur(transformed, (kernel_size, kernel_size), 0)
    
    # Random dilation/erosion
    if random.random() > 0.7:
        kernel_size = random.randint(1, 2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if random.random() > 0.5:
            transformed = cv2.dilate(transformed, kernel, iterations=1)
        else:
            transformed = cv2.erode(transformed, kernel, iterations=1)
    
    # Random brightness adjustment
    if random.random() > 0.5:
        alpha = random.uniform(0.8, 1.2)  # Brightness factor
        transformed = cv2.convertScaleAbs(transformed, alpha=alpha, beta=0)
    
    return transformed


def main():
    """Main function for generating obfuscated character images."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load character mapping
    char_mapping = load_character_mapping(args.char_mapping_path)
    if not char_mapping:
        logger.error("Could not load character mapping. Exiting.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each split (train, test)
    for split in ['train', 'test']:
        regular_split_dir = os.path.join(args.regular_dir, split)
        output_split_dir = os.path.join(args.output_dir, split)
        
        if not os.path.exists(regular_split_dir):
            logger.warning(f"Directory not found: {regular_split_dir}")
            continue
        
        os.makedirs(output_split_dir, exist_ok=True)
        
        # Process each character directory
        for char_dir in os.listdir(regular_split_dir):
            char_path = os.path.join(regular_split_dir, char_dir)
            if not os.path.isdir(char_path):
                continue
            
            # Create corresponding output directory
            output_char_dir = os.path.join(output_split_dir, char_dir)
            os.makedirs(output_char_dir, exist_ok=True)
            
            # Get list of image files
            image_files = [f for f in os.listdir(char_path) if f.endswith('.png')]
            
            # Limit the number of samples if specified
            if args.num_samples_per_char and len(image_files) > args.num_samples_per_char:
                image_files = random.sample(image_files, args.num_samples_per_char)
            
            logger.info(f"Processing {len(image_files)} images for character '{char_dir}' in {split} split")
            
            # Process each image
            for img_file in tqdm(image_files, desc=f"Char '{char_dir}'"):
                img_path = os.path.join(char_path, img_file)
                
                # Load the image
                try:
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        logger.warning(f"Could not load image: {img_path}")
                        continue
                except Exception as e:
                    logger.error(f"Error loading image {img_path}: {e}")
                    continue
                
                # Apply random transformations to create an "obfuscated" version
                obfuscated_img = apply_random_transformation(image)
                
                # Save the obfuscated image with the same filename
                output_path = os.path.join(output_char_dir, img_file)
                cv2.imwrite(output_path, obfuscated_img)
            
            logger.info(f"Generated {len(image_files)} obfuscated images for character '{char_dir}'")
    
    logger.info(f"Obfuscated image generation completed, saved to {args.output_dir}")


if __name__ == "__main__":
    main() 
