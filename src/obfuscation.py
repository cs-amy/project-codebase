import random
import re

# Define a dictionary for character substitution.
SUBSTITUTIONS = {
    'a': ['@', '4'],
    'e': ['3'],
    'i': ['!', '1'],
    'o': ['0'],
    's': ['$', '5'],
    'l': ['1', '|'],
    't': ['7']
}

def obfuscate_substitution(word: str) -> str:
    """
    Generate an obfuscated variant of a word by substituting characters based on SUBSTITUTIONS.
    Each character in the word is replaced with one of its substitutes.
    """
    obfuscated = []
    for char in word:
        lower_char = char.lower()
        if lower_char in SUBSTITUTIONS:
            # Choose a random substitute; preserve original case.
            substitute = random.choice(SUBSTITUTIONS[lower_char])
            # If the original char was uppercase, convert substitute to uppercase.
            obfuscated.append(substitute.upper() if char.isupper() else substitute)
        else:
            obfuscated.append(char)
    return ''.join(obfuscated)

def obfuscate_spaces(word: str) -> str:
    """
    Generate an obfuscated variant of a word by randomly inserting spaces between characters.
    For each gap between characters, insert a space with a probability of 90%.
    """
    if len(word) < 2:
        return word  # No spaces to insert in single-character strings.
    obfuscated = [word[0]]
    for char in word[1:]:
        # With 90% chance, add a space before the character.
        if random.random() < 0.9:
            obfuscated.append(" ")
        obfuscated.append(char)
    return ''.join(obfuscated)

def obfuscate_combined(word: str) -> str:
    """
    Generate an obfuscated variant that applies both substitution and space insertion.
    First, substitute characters, then insert spaces.
    """
    substituted = obfuscate_substitution(word)
    return obfuscate_spaces(substituted)

def generate_obfuscations(word: str) -> list:
    """
    Given an unobfuscated word, generate a list of obfuscated variants using 
    the substitution, space insertion, and combination techniques.
    
    Returns:
        A list containing three variants:
            - Variant with substitution only.
            - Variant with space insertion only.
            - Variant with both substitution and space insertion.
    """
    variant_substitution = obfuscate_substitution(word)
    variant_spaces = obfuscate_spaces(word)
    variant_combined = obfuscate_combined(word)
    return [variant_substitution, variant_spaces, variant_combined]
