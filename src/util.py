def read_entries_from_file(filepath):
    """Reads entries from a file (one per line) and returns a list of entries."""
    if not os.path.exists(filepath):
        print(f"File {filepath} not found!")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def read_fonts_from_folder(folder_path):
    """Reads all TTF files from the specified folder and returns a list of font paths."""
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found!")
        return []
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith('.ttf')]
