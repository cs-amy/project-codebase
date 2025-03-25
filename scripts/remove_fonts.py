"""
Script to remove obfuscated character images that use specific fonts.
This script searches through the obfuscated dataset and deletes images
that contain any of the specified font names in their filenames.
"""

import os
import sys
import argparse
import logging
import re
from pathlib import Path

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
        description="Remove obfuscated character images that use specific fonts")
    
    parser.add_argument("--data_dir", type=str, default=os.path.join(project_root, "data/characters/obfuscated"),
                       help="Directory containing the obfuscated character images")
    parser.add_argument("--fonts_to_remove", type=str, nargs='+',
                       default=[
                           "Arial_Rounded_MT_Bold", "Arial", "Baskerville_Old_Face",
                           "Bookman_Old_Style", "BrushScript_MT", "Freestyle_Script",
                           "French_Script_MT", "Georgia", "Microsoft_Tai_Le",
                           "mongolian_baiti", "monotype_corsiva", "segoe_script",
                           "segoe_ui_symbol", "serif"
                       ],
                       help="List of font names to remove (spaces replaced with underscores)")
    
    return parser.parse_args()


def normalize_font_name(font_name):
    """
    Normalize a font name for comparison.
    Removes spaces, converts to lowercase, and removes special characters.
    
    Args:
        font_name: Font name to normalize
        
    Returns:
        Normalized font name
    """
    # Replace spaces and underscores with a standard separator
    normalized = font_name.replace(' ', '_').replace('-', '_')
    # Convert to lowercase
    normalized = normalized.lower()
    # Remove special characters
    normalized = re.sub(r'[^\w_]', '', normalized)
    return normalized


def find_and_remove_font_images(data_dir, fonts_to_remove):
    """
    Find and remove images that use the specified fonts.
    
    Args:
        data_dir: Directory containing the dataset
        fonts_to_remove: List of font names to remove
        
    Returns:
        Total number of files removed
    """
    total_removed = 0
    
    # Normalize font names for comparison
    normalized_fonts = [normalize_font_name(font) for font in fonts_to_remove]
    
    # Process both train and test directories
    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            logger.warning(f"Directory not found: {split_dir}")
            continue
        
        # Process each character directory
        for char in os.listdir(split_dir):
            char_dir = os.path.join(split_dir, char)
            if not os.path.isdir(char_dir):
                continue
            
            files_to_remove = []
            # Check each image file
            for filename in os.listdir(char_dir):
                if not filename.endswith('.png'):
                    continue
                
                # Normalize the filename for comparison
                normalized_filename = normalize_font_name(filename)
                
                # Check if any of the fonts to remove is in the filename
                for font in normalized_fonts:
                    if font in normalized_filename:
                        files_to_remove.append(filename)
                        break
            
            # Remove the files
            for filename in files_to_remove:
                file_path = os.path.join(char_dir, filename)
                try:
                    os.remove(file_path)
                    total_removed += 1
                except Exception as e:
                    logger.error(f"Error removing file {file_path}: {e}")
            
            if files_to_remove:
                logger.info(f"Removed {len(files_to_remove)} files from {char_dir}")
    
    return total_removed


def main():
    """Main function."""
    args = parse_args()
    
    logger.info(f"Searching for images with specified fonts in {args.data_dir}")
    logger.info(f"Fonts to remove: {args.fonts_to_remove}")
    
    # Find and remove images
    total_removed = find_and_remove_font_images(args.data_dir, args.fonts_to_remove)
    
    logger.info(f"Removal complete. Total files removed: {total_removed}")


if __name__ == "__main__":
    main() 