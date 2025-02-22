import os
import random
import unittest
import tempfile
from PIL import Image

from src.image_generation import (
    clean_filename,
    read_entries_from_file,
    read_fonts_from_folder,
    create_text_image,
    generate_case_variants,
    generate_profanity_images
)
from src.obfuscation import (
    obfuscate_substitution,
    obfuscate_spaces,
    obfuscate_combined,
    generate_obfuscations,
)

class TestImageGeneration(unittest.TestCase):
    def setUp(self):
        # Set a fixed random seed for reproducibility
        random.seed(42)

    def test_clean_filename(self):
        self.assertEqual(clean_filename("n!qq3r"), "nqq3r")
        self.assertEqual(clean_filename("f**k"), "fk")
        self.assertEqual(clean_filename("sh!t"), "sht")

    def test_read_entries_from_file(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write("n!qq3r\nf**k\nsh!t\n")
            temp_file_name = temp_file.name
        entries = read_entries_from_file(temp_file_name)
        self.assertEqual(entries, ["n!qq3r", "f**k", "sh!t"])
        os.remove(temp_file_name)

    def test_read_fonts_from_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy TTF files and one non-TTF file.
            filenames = ["arial.ttf", "times.ttf", "not_a_font.txt"]
            for fname in filenames:
                with open(os.path.join(temp_dir, fname), 'w') as f:
                    f.write("dummy content")
            fonts = read_fonts_from_folder(temp_dir)
            # Should only include TTF files.
            self.assertEqual(len(fonts), 2)
            self.assertTrue(any("arial.ttf" in path.lower() for path in fonts))
            self.assertTrue(any("times.ttf" in path.lower() for path in fonts))

    def test_create_text_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_image.png")
            create_text_image("test", font_path="", output_path=output_path)
            self.assertTrue(os.path.exists(output_path))
            with Image.open(output_path) as img:
                self.assertEqual(img.format, "PNG")

    def test_obfuscation_routines(self):
        word = "test"
        sub_variant = obfuscate_substitution(word)
        spaces_variant = obfuscate_spaces(word)
        combined_variant = obfuscate_combined(word)
        variants = generate_obfuscations(word)
        # Ensure the substitution variant has the same length as original.
        self.assertEqual(len(sub_variant), len(word))
        # Removing spaces from spaces_variant should equal original.
        self.assertEqual(spaces_variant.replace(" ", "").lower(), word.lower())
        # Combined variant should be different.
        self.assertNotEqual(combined_variant.replace(" ", "").lower(), word.lower())
        # generate_obfuscations should return exactly three variants.
        self.assertEqual(len(variants), 3)

    def test_generate_case_variants(self):
        word = "TestWord"
        variants = generate_case_variants(word)
        self.assertEqual(len(variants), 4)
        # Check that one variant is the original.
        self.assertIn(word, variants)
        # Check for lowercase and uppercase.
        self.assertIn(word.lower(), variants)
        self.assertIn(word.upper(), variants)
        # Mixed-case should differ from both full lowercase and uppercase.
        mixed = [v for v in variants if v not in (word, word.lower(), word.upper())]
        self.assertTrue(len(mixed) == 1)
        self.assertNotEqual(mixed[0], word.lower())
        self.assertNotEqual(mixed[0], word.upper())

    def test_generate_profanity_images(self):
        # Create temporary files for profanities and a fonts folder.
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary profanities file.
            prof_file = os.path.join(temp_dir, "profanities.txt")
            with open(prof_file, "w") as f:
                f.write("n!qq3r\nf**k\n")
            
            # Create a temporary fonts folder with one dummy TTF file.
            fonts_folder = os.path.join(temp_dir, "fonts")
            os.makedirs(fonts_folder, exist_ok=True)
            dummy_font_path = os.path.join(fonts_folder, "arial.ttf")
            with open(dummy_font_path, "w") as f:
                f.write("dummy font content")
            
            # Create an output directory.
            output_dir = os.path.join(temp_dir, "output_images")
            os.makedirs(output_dir, exist_ok=True)
            
            # Run the image generation with multi_cases and include obfuscations.
            generate_profanity_images(profanities_file=prof_file, fonts_folder=fonts_folder, 
                                          output_dir=output_dir, include_obfuscations=True, multi_cases=True)
            
            # Check that images have been generated.
            generated_files = os.listdir(output_dir)
            # For each profanity word, with multi-case (4 variants) and 1 plain + 3 obfuscated images per case,
            # Total images per word = 4 * (1 + 3) = 16. With 2 words, expect 32 images.
            self.assertEqual(len(generated_files), 32)
            
            # Verify that each generated file is a PNG.
            for file in generated_files:
                with Image.open(os.path.join(output_dir, file)) as img:
                    self.assertEqual(img.format, "PNG")

if __name__ == '__main__':
    unittest.main()