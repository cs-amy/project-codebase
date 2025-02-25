import os
import random
from PIL import Image, ImageDraw, ImageFont # type: ignore
from src.obfuscation import generate_obfuscations
from src.util import (
    clean_filename,
    read_entries_from_file, 
    read_fonts_from_folder
)


def create_text_image(text, font_path, font_size=12, padding=20, bg_color='white', text_color='black', output_path='output.png'):
    """
    Generates an image with a single word (text) displayed using the specified font.
    """
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            print(f"Font at {font_path} not found; using default font.")
            font = ImageFont.load_default()
    except IOError:
        print(f"Error loading font from {font_path}; using default font.")
        font = ImageFont.load_default()
    
    dummy_img = Image.new('RGB', (1, 1), color=bg_color)
    draw = ImageDraw.Draw(dummy_img)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]  # right - left
    text_height = text_bbox[3] - text_bbox[1]  # bottom - top
    img_width = text_width + 2 * padding
    img_height = text_height + 2 * padding
    img = Image.new('RGB', (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((padding, padding), text, font=font, fill=text_color)
    
    img.save(output_path)
    print(f"Saved image: {output_path}")

def generate_case_variants(word: str) -> list:
    """
    Generates different case versions of a word:
      - Original (as-is)
      - Lowercase
      - Uppercase
      - Mixed-case (random mix of upper and lower)
      
    Returns a list of these four variants.
    """
    lower = word.lower()
    upper = word.upper()
    mixed = ''.join(random.choice([char.lower(), char.upper()]) for char in word)
    return [word, lower, upper, mixed]

def generate_profanity_images(profanities_file, fonts_folder, output_dir, include_obfuscations=True, multi_cases=True):
    """
    Generates images for each profanity from the profanities file using each typeface from the fonts folder.
    
    If multi_cases is True, generates images for different case variants of each profanity.
    For each word (or case variant), an image of the plain text is generated.
    If include_obfuscations is True, additional images for each obfuscated variant are generated.
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
                # Generate image for the plain variant (white and black backgrounds).
                output_file_plain = os.path.join(output_dir, f"{cleaned_variant}_{font_name}.png")
                create_text_image(variant_word, bg_color='white', text_color='black', font_path=font_path, output_path=f"{output_file_plain}_white.png")
                create_text_image(variant_word, bg_color='black', text_color='white', font_path=font_path, output_path=f"{output_file_plain}_black.png")
                
                # If obfuscations are enabled, generate and save obfuscated variants (white and black backgrounds).
                if include_obfuscations:
                    obf_variants = generate_obfuscations(variant_word)
                    for idx, obf_variant in enumerate(obf_variants, start=1):
                        output_file_variant = os.path.join(output_dir, f"{cleaned_variant}_{font_name}_variant{idx}.png")
                        create_text_image(obf_variant, bg_color='white', text_color='black', font_path=font_path, output_path=f"{output_file_variant}_white.png")
                        create_text_image(obf_variant, bg_color='black', text_color='white', font_path=font_path, output_path=f"{output_file_variant}_black.png")
