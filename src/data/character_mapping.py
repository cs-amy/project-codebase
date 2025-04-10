"""
Character mapping module defining relationships between standard characters and their obfuscated variants.
"""

from typing import Dict, List


# Dictionary mapping standard characters to their common obfuscated variants
CHAR_TO_OBFUSCATED: Dict[str, List[str]] = {
    # Lowercase letters
    'a': ['@', 'α', '4', 'а', 'ɑ', 'ä', 'á', 'à', 'â', 'ă', 'å'],
    'b': ['ƅ', '6', 'б', 'β', 'ь', 'ß', 'þ'],
    'c': ['©', '¢', 'ç', 'с', 'ć', 'č'],
    'd': ['đ', 'ð', 'ď', 'δ', 'ɗ'],
    'e': ['3', '€', 'ε', 'έ', 'е', 'ë', 'é', 'è', 'ê', 'ě', 'ĕ'],
    'f': ['ƒ', 'ƭ', 'φ', 'ф'],
    'g': ['9', 'ğ', 'ģ', 'ǥ', 'γ'],
    'h': ['һ', 'ħ', 'ɦ', 'н'],
    'i': ['1', '!', '|', 'ı', 'ί', 'и', 'î', 'ï', 'í', 'ì', 'į'],
    'j': ['ј', 'ʝ'],
    'k': ['κ', 'к', 'ķ'],
    'l': ['1', '|', 'ł', 'ι', 'ℓ', 'ĺ', 'л'],
    'm': ['м', 'ɱ', 'ḿ'],
    'n': ['ո', 'η', 'ñ', 'ń', 'н'],
    'o': ['0', 'ο', 'ө', 'ø', 'ö', 'ó', 'ò', 'ô', 'о'],
    'p': ['ρ', 'р', 'þ'],
    'q': ['ԛ', 'φ', 'ʠ'],
    'r': ['®', 'ř', 'ŕ', 'г', 'я'],
    's': ['5', '$', 'ѕ', 'ś', 'š'],
    't': ['τ', 'т', 'ţ', 'ť'],
    'u': ['υ', 'ц', 'µ', 'ü', 'ú', 'ù', 'û'],
    'v': ['ν', 'ѵ', 'υ'],
    'w': ['ω', 'ѡ', 'ψ', 'ώ', 'ш', 'щ'],
    'x': ['×', 'χ', 'ж', 'х'],
    'y': ['ү', 'γ', 'у', 'ý', 'ÿ'],
    'z': ['ʐ', 'ż', 'ź', 'ž', 'з'],
    
    # Uppercase letters
    'A': ['Α', '4', 'Д', 'Ä', 'Á', 'À', 'Â'],
    'B': ['8', 'β', 'Β', 'В'],
    'C': ['Ç', 'Ć', 'Č', 'С'],
    'D': ['Ð', 'Ď'],
    'E': ['3', 'Σ', 'Έ', 'Ε', 'Е', 'Ë', 'É', 'È', 'Ê'],
    'F': ['Φ', 'Ƒ'],
    'G': ['6', 'Ğ', 'Ģ', 'Γ'],
    'H': ['Η', 'Н'],
    'I': ['1', '|', 'Í', 'Ì', 'Î', 'Ï', 'И'],
    'J': ['Ј'],
    'K': ['Κ', 'К'],
    'L': ['Ι', 'Ł', 'Ĺ', 'Л'],
    'M': ['Μ', 'М'],
    'N': ['Ν', 'Ń', 'Ñ', 'Н'],
    'O': ['0', 'Θ', 'Ο', 'Ө', 'Ø', 'Ö', 'Ó', 'Ò', 'Ô'],
    'P': ['Ρ', 'Р'],
    'Q': ['Φ'],
    'R': ['®', 'Я', 'Ř', 'Ŕ'],
    'S': ['5', '$', 'Ѕ', 'Ś', 'Š'],
    'T': ['Τ', 'Т'],
    'U': ['Υ', 'Ц', 'Ü', 'Ú', 'Ù', 'Û'],
    'V': ['Ѵ', 'V'],
    'W': ['Ω', 'Ѡ', 'Ψ', 'Ш', 'Щ'],
    'X': ['Χ', 'Ж', 'Х'],
    'Y': ['Υ', 'Ү', 'Ý', 'Ÿ'],
    'Z': ['Ζ', 'Ż', 'Ź', 'Ž', 'З'],
    
    # Numbers
    '0': ['o', 'O', 'ο', 'О', 'ø', 'Ø'],
    '1': ['l', 'I', 'i', '|', 'L'],
    '2': ['Z', 'z', 'ƻ'],
    '3': ['E', 'ε', 'Є', 'з'],
    '4': ['A', 'a', 'λ'],
    '5': ['S', 's', 'ς'],
    '6': ['b', 'б', 'ó'],
    '7': ['T', 'τ'],
    '8': ['B', 'β'],
    '9': ['g', 'q', 'ԛ', 'ğ'],
}


# Create reverse mapping (obfuscated to standard)
OBFUSCATED_TO_CHAR: Dict[str, str] = {}
for char, obfuscated_list in CHAR_TO_OBFUSCATED.items():
    for obfuscated in obfuscated_list:
        OBFUSCATED_TO_CHAR[obfuscated] = char


def get_obfuscated_variants(char: str) -> List[str]:
    """
    Get list of possible obfuscated variants for a character.
    
    Args:
        char: Standard character
        
    Returns:
        List of obfuscated variants
    """
    if char in CHAR_TO_OBFUSCATED:
        return CHAR_TO_OBFUSCATED[char]
    return []


def get_standard_character(obfuscated: str) -> str:
    """
    Get the standard character for an obfuscated variant.
    
    Args:
        obfuscated: Obfuscated character
        
    Returns:
        Standard character if found, otherwise the original character
    """
    return OBFUSCATED_TO_CHAR.get(obfuscated, obfuscated)


def is_obfuscated(char: str) -> bool:
    """
    Check if a character is an obfuscated variant.
    
    Args:
        char: Character to check
        
    Returns:
        True if the character is an obfuscated variant, False otherwise
    """
    return char in OBFUSCATED_TO_CHAR


def get_all_standard_characters() -> List[str]:
    """
    Get a list of all standard characters.
    
    Returns:
        List of all standard characters
    """
    return list(CHAR_TO_OBFUSCATED.keys())


def get_all_obfuscated_characters() -> List[str]:
    """
    Get a list of all obfuscated characters.
    
    Returns:
        List of all obfuscated characters
    """
    return list(OBFUSCATED_TO_CHAR.keys())


def deobfuscate_text(text: str) -> str:
    """
    Convert obfuscated text to standard text.
    
    Args:
        text: Obfuscated text
        
    Returns:
        Deobfuscated text
    """
    return ''.join(get_standard_character(char) for char in text)


def obfuscate_text(text: str, obfuscation_level: float = 0.7) -> str:
    """
    Convert standard text to obfuscated text with a given obfuscation level.
    
    Args:
        text: Standard text
        obfuscation_level: Probability of obfuscating each character (0.0 to 1.0)
        
    Returns:
        Obfuscated text
    """
    import random
    
    obfuscated = []
    for char in text:
        if char in CHAR_TO_OBFUSCATED and random.random() < obfuscation_level:
            variants = CHAR_TO_OBFUSCATED[char]
            obfuscated.append(random.choice(variants))
        else:
            obfuscated.append(char)
    
    return ''.join(obfuscated) 