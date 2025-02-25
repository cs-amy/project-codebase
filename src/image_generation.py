import os
import random
import numpy as np # type: ignore
from typing import List
from PIL import Image, ImageDraw, ImageFont # type: ignore
from src.obfuscation import generate_obfuscations
from src.util import (
    clean_filename,
    read_entries_from_file, 
    read_fonts_from_folder
)

def create_noisy_background(width: int, height: int) -> Image.Image:
    """
    Generate a noisy RGB background of the specified width and height.
    Each pixel's RGB value is randomly chosen between 0 and 255.

    Parameters:
        width: int - The width of the background.
        height: int - The height of the background.

    Returns:
        The noisy background image.
    """
    # Create a random 3D array (height, width, 3) with values in [0, 255].
    noise_array = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(noise_array, mode='RGB')

def create_text_image(
    text: str, 
    font_path: str = None, 
    font_size: int = 12,
    padding: int = 20, 
    bg_color: str = 'white', 
    text_color: str = 'black', 
    output_path: str = 'output.png'
) -> None:
    """
    Generates an image with a single word (text) displayed using the specified font.
    
    Parameters:
        text: str - The word to display in the image.
        font_path: str - The path to the font file to use.
        font_size: int - The font size to use.
        padding: int - The padding to use around the text.
        bg_color: str - The background color of the image.
        text_color: str - The color of the text.
        output_path: str - The path to save the image.
    """
    # Attempt to load the specified font, fall back to default if not found.
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            print(f"Font at {font_path} not found; using default font.")
            font = ImageFont.load_default()
    except IOError:
        print(f"Error loading font from {font_path}; using default font.")
        font = ImageFont.load_default()
    
    # Create a dummy image to measure text size.
    dummy_img = Image.new('RGB', (1, 1), color=bg_color)
    draw = ImageDraw.Draw(dummy_img)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Define final image dimensions based on text size and padding.
    img_width = text_width + 2 * padding
    img_height = text_height + 2 * padding

    # Create a plain background image with the specified dimensions.
    img = Image.new('RGB', (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((padding, padding), text, font=font, fill=text_color)

    # Create a noisy background image with the specified dimensions.
    noisy_img = create_noisy_background(img_width, img_height)
    draw = ImageDraw.Draw(noisy_img)
    draw.text((padding, padding), text, font=font, fill=text_color)
    
    # Save the final images.
    img.save(output_path)
    noisy_img.save(f"{output_path}_noisy.png")
    print(f"Saved image (plain): {output_path}")
    print(f"Saved image (noisy): {output_path}_noisy.png")

def generate_case_variants(word: str) -> list[str]:
    """
    Generates different case versions of a word:
      - Original (as-is)
      - Lowercase
      - Uppercase
      - Mixed-case (random mix of upper and lower)
    
    Parameters:
        word: str - The word to generate case variants for.
      
    Returns a list of these four variants.
    """
    lower = word.lower()
    upper = word.upper()
    mixed = ''.join(random.choice([char.lower(), char.upper()]) for char in word)
    return [word, lower, upper, mixed]

def generate_profanity_images(
    profanities_file: str, 
    fonts_folder: str, 
    output_dir: str, 
    include_obfuscations: bool = True, 
    multi_cases: bool = True,
    font_sizes: List[int] = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
) -> None:
    """
    Generates images for each profanity from the profanities file using each typeface from the fonts folder.
    
    If multi_cases is True, generates images for different case variants of each profanity.
    For each word (or case variant), an image of the plain text is generated.
    If include_obfuscations is True, additional images for each obfuscated variant are generated.

    Parameters:
        profanities_file: str - The path to the profanities file.
        fonts_folder: str - The path to the fonts folder.
        output_dir: str - The path to save the images.
        include_obfuscations: bool - Whether to include obfuscated variants.
        multi_cases: bool - Whether to include different case variants.
        font_sizes: List[int] - A list of font sizes to try.
    """
    profanity_list = read_entries_from_file(profanities_file)
    font_paths = read_fonts_from_folder(fonts_folder)
    
    if not profanity_list:
        print("No profanities found. Check the profanities file.")
        return
    if not font_paths:
        print("No fonts found in the specified folder.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for word in profanity_list:
        # If multi_cases is enabled, generate different case variants.
        case_variants = generate_case_variants(word) if multi_cases else [word]
        
        for variant_word in case_variants:
            cleaned_variant = clean_filename(variant_word)
            for font_path in font_paths:
                font_name = clean_filename(os.path.splitext(os.path.basename(font_path))[0])
                font_size = random.choice(font_sizes)
                # Generate image for the plain variant (white and black backgrounds).
                output_file_plain = os.path.join(output_dir, f"{cleaned_variant}_{font_name}_{font_size}.png")
                create_text_image(
                    variant_word, 
                    bg_color='white', 
                    text_color='black', 
                    font_path=font_path, 
                    font_size=font_size, 
                    output_path=output_file_plain
                )
                create_text_image(
                    variant_word, 
                    bg_color='black', 
                    text_color='white', 
                    font_path=font_path, 
                    font_size=font_size, 
                    output_path=output_file_plain
                )
                
                # If obfuscations are enabled, generate and save obfuscated variants (white and black backgrounds).
                if include_obfuscations:
                    obf_variants = generate_obfuscations(variant_word)
                    for idx, obf_variant in enumerate(obf_variants, start=1):
                        output_file_variant = os.path.join(output_dir, f"{cleaned_variant}_{font_name}_variant{idx}_{font_size}.png")
                        create_text_image(
                            obf_variant, 
                            bg_color='white', 
                            text_color='black', 
                            font_path=font_path, 
                            font_size=font_size, 
                            output_path=output_file_variant
                        )
                        create_text_image(
                            obf_variant, 
                            bg_color='black', 
                            text_color='white', 
                            font_path=font_path, 
                            font_size=font_size, 
                            output_path=output_file_variant
                        )
