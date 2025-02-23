import os
import re
from typing import List

def clean_filename(s: str) -> str:
    """Clean string for safe filename usage by removing non-alphanumeric characters."""
    return re.sub(r'\W+', '', s)

def read_entries_from_file(filepath: str) -> List[str]:
    """Reads entries from a file (one per line) and returns a list of entries."""
    if not os.path.exists(filepath):
        print(f"File {filepath} not found!")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def read_fonts_from_folder(folder_path: str) -> List[str]:
    """Reads all TTF files from the specified folder and returns a list of font paths."""
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found!")
        return []
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith('.ttf')]
