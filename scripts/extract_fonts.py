"""
Script to extract and compile a list of all fonts used in the regular character dataset.
This script parses the filenames of images in the regular character dataset,
extracts font information, and saves a list of unique fonts to a file.
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path
from collections import Counter

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
        description="Extract and list all fonts used in the regular character dataset")
    
    parser.add_argument("--data_dir", type=str, default="data/characters/regular/train",
                       help="Directory containing the regular character images")
    parser.add_argument("--output_file", type=str, default="fonts.txt",
                       help="Output file to save the font list")
    
    return parser.parse_args()


def extract_font_name(filename):
    """
    Extract the font name from a filename.
    The function assumes that filenames contain the font name followed by a number.
    
    Args:
        filename: The image filename
        
    Returns:
        Extracted font name or None if no font name is found
    """
    # Remove file extension
    basename = os.path.splitext(filename)[0]
    
    # Extract font name (assuming format is FontName12345.png)
    # This regex matches a name (letters, spaces, and some special chars) followed by digits
    match = re.match(r'([A-Za-z0-9\s\-_.]+?)(\d+)$', basename)
    
    if match:
        font_name = match.group(1).strip()
        return font_name
    
    return None


def scan_directory(directory):
    """
    Scan a directory recursively and extract font names from filenames.
    
    Args:
        directory: The directory to scan
        
    Returns:
        Counter of font names and their frequencies
    """
    font_counter = Counter()
    
    # Walk through all subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                font_name = extract_font_name(file)
                if font_name:
                    font_counter[font_name] += 1
    
    return font_counter


def main():
    """Main function."""
    args = parse_args()
    
    logger.info(f"Scanning directory: {args.data_dir}")
    
    # Scan the dataset directory
    font_counter = scan_directory(args.data_dir)
    
    # Sort fonts by frequency (most common first)
    sorted_fonts = font_counter.most_common()
    
    # Prepare the output
    total_fonts = len(sorted_fonts)
    total_images = sum(font_counter.values())
    
    logger.info(f"Found {total_fonts} unique fonts in {total_images} images")
    
    # Save the font list to a file
    output_path = os.path.join(project_root, args.output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# List of fonts used in the regular character dataset\n")
        f.write(f"# Total unique fonts: {total_fonts}\n")
        f.write(f"# Total images analyzed: {total_images}\n\n")
        
        # Write only font names, one per line
        for font_name, _ in sorted_fonts:
            f.write(f"{font_name}\n")
    
    logger.info(f"Font list saved to {output_path}")


if __name__ == "__main__":
    main() 