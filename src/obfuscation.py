import random
from typing import Dict, List

# Define a dictionary for character substitution.
SUBSTITUTIONS: Dict[str, List[str]] = {
    'a': ['@', '4'],
    'b': ['6'],
    'c': ['¢'],
    'e': ['3'],
    'g': ['9', 'q'],
    'i': ['!'],
    'l': ['1'],
    'o': ['0'],
    'p': ['p'],
    'q': ['9'],
    's': ['$', '5'],
    't': ['7'],
    'z': ['2']
}

def obfuscate_substitution(word: str) -> str:
    """
    Generate an obfuscated variant of a word by substituting characters based on SUBSTITUTIONS.
    Each character in the word is replaced with one of its substitutes.

    Parameters:
        word: str - The word to obfuscate.

    Returns:
        The obfuscated word.
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

    Parameters:
        word: str - The word to obfuscate.

    Returns:
        The obfuscated word.
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

    Parameters:
        word: str - The word to obfuscate.

    Returns:
        The obfuscated word.
    """
    substituted = obfuscate_substitution(word)
    return obfuscate_spaces(substituted)

def generate_obfuscations(word: str) -> list[str]:
    """
    Given an unobfuscated word, generate a list of obfuscated variants using 
    the substitution, space insertion, and combination techniques.

    Parameters:
        word: str - The word to obfuscate.

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
