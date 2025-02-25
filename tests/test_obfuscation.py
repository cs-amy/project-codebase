import unittest
import random
from src.obfuscation import (
    obfuscate_substitution,
    obfuscate_spaces,
    obfuscate_combined,
    generate_obfuscations
)

class TestObfuscation(unittest.TestCase):
    def setUp(self):
        # Set a fixed random seed for reproducibility in tests.
        random.seed(3)

    def test_obfuscate_substitution(self):
        word = "test"
        result = obfuscate_substitution(word)
        # Check that the result is a string and has the same length as the original.
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), len(word))
        # Allowed substitutions for our test word "test":
        # For 't': either 't' or '7'
        # For 'e': either 'e' or '3'
        # For 's': either 's', '$', or '5'
        allowed_substitutions = {
            't': ['t', '7'],
            'e': ['e', '3'],
            's': ['s', '$', '5']
        }
        for orig, out in zip(word.lower(), result.lower()):
            if orig in allowed_substitutions:
                self.assertIn(out, allowed_substitutions[orig],
                              f"Character '{orig}' was substituted with '{out}' which is not in {allowed_substitutions[orig]}")
            else:
                self.assertEqual(orig, out)

    def test_obfuscate_spaces(self):
        word = "test"
        result = obfuscate_spaces(word)
        # Ensure result is a string and at least as long as the original word.
        self.assertIsInstance(result, str)
        self.assertGreaterEqual(len(result), len(word))
        # Removing spaces should yield the original word.
        result_no_spaces = result.replace(" ", "")
        self.assertEqual(result_no_spaces.lower(), word.lower())

    def test_obfuscate_combined(self):
        word = "test"
        result = obfuscate_combined(word)
        self.assertIsInstance(result, str)
        # The combined variant should differ from the original after removing spaces.
        result_no_spaces = result.replace(" ", "")
        # It is possible (though unlikely) that substitution doesn't change the word;
        # however, the space insertion should cause a difference.
        self.assertNotEqual(result_no_spaces.lower(), word.lower(),
                            "Combined obfuscation did not alter the word as expected.")

    def test_generate_obfuscations(self):
        word = "test"
        results = generate_obfuscations(word)
        # The function should return exactly three variants.
        self.assertEqual(len(results), 3)
        for variant in results:
            self.assertIsInstance(variant, str)
            self.assertTrue(len(variant) > 0, "Variant should not be an empty string.")

if __name__ == '__main__':
    unittest.main()
