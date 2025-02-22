import os
import random
import unittest
import tempfile
from PIL import Image
from src.util import (
    clean_filename,
    read_entries_from_file,
    read_fonts_from_folder
)


class TestUtil(unittest.TestCase):
    def setUp(self):
        # Set a fixed random seed for reproducibility
        random.seed(2)

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

if __name__ == '__main__':
    unittest.main()