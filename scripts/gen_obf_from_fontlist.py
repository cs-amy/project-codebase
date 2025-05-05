"""
Script for generating obfuscated character images using fonts from fonts/fonts_obfuscated.txt.
This script creates one image per obfuscated variant per font in the fonts/fonts_obfuscated.txt file.
"""

import os
import sys
import random
import argparse
import logging
import yaml
from pathlib import Path
import platform
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate obfuscated character variants with fonts from fonts/fonts_obfuscated.txt")

    parser.add_argument("--output_dir", type=str, default="data/characters/obfuscated/train",
                        help="Path to output directory for obfuscated images")
    parser.add_argument("--char_mapping_path", type=str, default="configs/character_mapping.yaml",
                        help="Path to character mapping YAML file")
    parser.add_argument("--font_list_path", type=str, default="fonts/fonts_obfuscated.txt",
                        help="Path to the font list file")
    parser.add_argument("--image_size", type=int, default=64,
                        help="Size of the output images (square)")
    parser.add_argument("--font_size", type=int, default=36,
                        help="Font size to use for rendering characters")
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
            mapping = {k: v for k, v in mapping.items() if isinstance(
                k, str) and isinstance(v, list)}
        return mapping
    else:
        logger.warning(f"Character mapping file not found at {mapping_path}")
        return None


def load_font_list(font_list_path):
    """
    Load the list of fonts from the font list file.
    Skips comments and empty lines.

    Args:
        font_list_path: Path to the font list file

    Returns:
        List of font names
    """
    fonts = []
    if not os.path.exists(font_list_path):
        logger.error(f"Font list file not found: {font_list_path}")
        return fonts

    with open(font_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            fonts.append(line)

    logger.info(f"Loaded {len(fonts)} fonts from {font_list_path}")
    return fonts


def find_system_fonts():
    """Find system fonts based on the operating system."""
    system = platform.system()
    font_paths = []

    if system == 'Darwin':  # macOS
        font_dirs = [
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts")
        ]
    elif system == 'Windows':
        font_dirs = [
            "C:\\Windows\\Fonts"
        ]
    else:  # Linux and others
        font_dirs = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts")
        ]

    # Common font extensions
    extensions = ['.ttf', '.otf', '.ttc', '.dfont']

    # Find all font files in the directories
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for root, _, files in os.walk(font_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        font_paths.append(os.path.join(root, file))

    return font_paths


def map_font_names_to_files(requested_fonts):
    """
    Map font names from the font list to actual font files.

    Args:
        requested_fonts: List of font names to match

    Returns:
        Dictionary mapping font names to font file paths
    """
    system_fonts = find_system_fonts()
    logger.info(f"Found {len(system_fonts)} font files on the system")

    font_map = {}
    for font_name in requested_fonts:
        matched = False
        # Look for an exact match first
        for font_path in system_fonts:
            filename = os.path.basename(font_path).lower()
            font_name_lower = font_name.lower()
            if font_name_lower in filename:
                font_map[font_name] = font_path
                matched = True
                break

        if not matched:
            # Look for partial matches if no exact match
            for font_path in system_fonts:
                # Split font name by spaces and look for parts
                parts = font_name.lower().split()
                filename = os.path.basename(font_path).lower()
                if any(part in filename for part in parts if len(part) > 2):
                    font_map[font_name] = font_path
                    matched = True
                    break

        if not matched:
            logger.warning(
                f"Could not find a matching font file for '{font_name}'")

    logger.info(
        f"Mapped {len(font_map)} out of {len(requested_fonts)} requested fonts")
    return font_map


def test_font_for_character(font_path, character):
    """Test if a font can properly render a character."""
    try:
        font = ImageFont.truetype(font_path, 20)
        img = Image.new('L', (30, 30), color=255)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), character, fill=0, font=font)
        # Check if the image actually contains something
        array = np.array(img)
        if np.sum(255 - array) < 10:  # Almost blank image
            return False
        return True
    except Exception:
        return False


def render_character(character, font_path, image_size=64, font_size=36):
    """
    Render a character as an image with specified font.

    Args:
        character: The character to render
        font_path: Path to the font file
        image_size: Size of the output image (square)
        font_size: Font size to use

    Returns:
        np.ndarray: Grayscale image of the rendered character
    """
    # Create a blank image with white background
    img = Image.new('L', (image_size, image_size), color=255)
    draw = ImageDraw.Draw(img)

    try:
        # Try to use the specified font
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            logger.debug(f"Error loading font {font_path}: {e}")
            font = ImageFont.load_default()

        # Calculate text position to center it
        try:
            text_width, text_height = font.getsize(character)
        except AttributeError:
            # For newer PIL versions
            text_width, text_height = font.getbbox(character)[2:4]

        x = (image_size - text_width) // 2
        y = (image_size - text_height) // 2

        # Draw the character
        draw.text((x, y), character, fill=0, font=font)

        # Convert to numpy array
        image_array = np.array(img)

        return image_array

    except Exception as e:
        logger.debug(
            f"Error rendering character '{character}' with font {font_path}: {e}")
        # Return a blank image in case of error
        return np.ones((image_size, image_size), dtype=np.uint8) * 255


def create_directory_structure(output_dir):
    """Create the directory structure for the dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Create directories for each letter (a-z)
    for char in 'abcdefghijklmnopqrstuvwxyz':
        char_dir = os.path.join(output_dir, char)
        os.makedirs(char_dir, exist_ok=True)


def generate_images(output_dir, char_mapping, font_map, image_size=64, font_size=36):
    """
    Generate obfuscated character images for each variant with each font.

    Args:
        output_dir: Output directory for the generated images
        char_mapping: Dictionary mapping standard characters to obfuscated variants
        font_map: Dictionary mapping font names to font file paths
        image_size: Size of output images
        font_size: Font size to use

    Returns:
        Total number of images generated
    """
    total_generated = 0
    total_expected = 0

    # For each lowercase letter
    for char in 'abcdefghijklmnopqrstuvwxyz':
        if char not in char_mapping:
            logger.warning(f"No mapping found for character '{char}'")
            continue

        variants = char_mapping[char]
        expected_count = len(variants) * len(font_map)
        total_expected += expected_count

        logger.info(
            f"Generating {len(variants)} variants with {len(font_map)} fonts each for character '{char}'")

        char_dir = os.path.join(output_dir, char)

        # For each variant
        for variant_idx, variant in enumerate(variants):
            # For each font
            for font_idx, (font_name, font_path) in enumerate(tqdm(font_map.items(), desc=f"Char '{char}' var {variant_idx+1}")):
                try:
                    # Check if this font can render this character
                    if not test_font_for_character(font_path, variant):
                        continue

                    # Render the variant character with this font
                    img = render_character(
                        variant,
                        font_path,
                        image_size=image_size,
                        font_size=font_size
                    )

                    # Determine file name for the variant
                    if any(c in variant for c in r'\/:|<>"?*'):
                        # For variants with special characters that might cause filename issues, use indices
                        output_path = os.path.join(
                            char_dir, f"variant_{variant_idx+1}_font_{font_idx+1}.png")
                    else:
                        safe_font_name = font_name.replace(' ', '_').replace(
                            ',', '').replace('(', '').replace(')', '')[:30]
                        output_path = os.path.join(
                            char_dir, f"variant_{variant_idx+1}_{variant}_{safe_font_name}.png")

                    # Save the image
                    cv2.imwrite(output_path, img)
                    total_generated += 1

                except Exception as e:
                    logger.error(
                        f"Error creating image for variant '{variant}' with font {font_name}: {e}")

        logger.info(f"Completed generation for character '{char}'")

    logger.info(
        f"Generated {total_generated} out of {total_expected} expected images")
    return total_generated


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

    # Load font list
    font_list = load_font_list(args.font_list_path)
    if not font_list:
        logger.error("No fonts found in the font list. Exiting.")
        return

    # Map font names to font files
    font_map = map_font_names_to_files(font_list)
    if not font_map:
        logger.error("Could not map any font names to font files. Exiting.")
        return

    # Create directory structure
    create_directory_structure(args.output_dir)

    # Generate images
    total_generated = generate_images(
        args.output_dir,
        char_mapping,
        font_map,
        args.image_size,
        args.font_size
    )

    logger.info(
        f"Obfuscated image generation completed. Generated {total_generated} images.")
    logger.info(f"Images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
