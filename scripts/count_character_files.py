"""
Script to count the number of files in each character directory 
for the regular and obfuscated datasets.
"""

import os
import string
import argparse
from tabulate import tabulate # type: ignore

def count_files_in_directory(base_path, folder_type, split_type):
    """
    Count files in each character directory (a-z) for the specified folder type and split.
    
    Args:
        base_path (str): Base path to the characters directory
        folder_type (str): Either 'regular' or 'obfuscated'
        split_type (str): Either 'train' or 'test'
    
    Returns:
        dict: Dictionary containing counts for each character directory
    """
    counts = {}
    folder_path = os.path.join(base_path, folder_type, split_type)
    
    # Check if directory exists
    if not os.path.exists(folder_path):
        print(f"Warning: Directory {folder_path} does not exist")
        return counts
    
    # Count files in each character directory
    for char in string.ascii_lowercase:
        char_dir = os.path.join(folder_path, char)
        if os.path.exists(char_dir):
            file_count = len([f for f in os.listdir(char_dir) if os.path.isfile(os.path.join(char_dir, f))])
            counts[char] = file_count
        else:
            counts[char] = 0
    
    return counts

def print_counts(counts_dict, split_type):
    """Print counts for a specific split type."""
    print(f"  - {split_type}:")
    subtotal = 0
    for char in string.ascii_lowercase:
        count = counts_dict[split_type][char]
        subtotal += count
        print(f"    - {char}: {count}")
    print(f"    Subtotal: {subtotal}")
    return subtotal

def print_table(counts_dict, folder_type):
    """Print counts in a table format."""
    # Prepare data for the table
    table_data = []
    train_subtotal = 0
    test_subtotal = 0
    
    for char in string.ascii_lowercase:
        train_count = counts_dict['train'][char]
        test_count = counts_dict['test'][char]
        train_subtotal += train_count
        test_subtotal += test_count
        table_data.append([char, train_count, test_count])
    
    # Add subtotals
    table_data.append(['Subtotal', train_subtotal, test_subtotal])
    
    # Print the table
    print(f"\n{folder_type.capitalize()} Dataset:")
    print(tabulate(table_data, 
                  headers=['Character', 'Train', 'Test'],
                  tablefmt='heavy_grid',
                  numalign='right',
                  colalign=('center', 'center', 'center')))
    return train_subtotal + test_subtotal

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Count and display statistics about character files in the dataset.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display in default format
  python count_character_files.py
  
  # Display in table format
  python count_character_files.py --table
        """
    )
    parser.add_argument('--table', action='store_true', 
                       help='Display results in a table format')
    args = parser.parse_args()
    
    # Print heading
    print("=" * 80)
    print("CHARACTER DATASET STATISTICS")
    print("=" * 80)
    
    # Get the absolute path to the project-codebase directory
    current_dir = os.getcwd()
    
    # Construct the absolute path to the characters directory
    characters_dir = os.path.join(current_dir, 'data', 'characters')
    
    # Count files for both regular and obfuscated folders
    grand_total = 0
    for folder_type in ['regular', 'obfuscated']:
        folder_counts = {}
        
        for split_type in ['train', 'test']:
            folder_counts[split_type] = count_files_in_directory(characters_dir, folder_type, split_type)
        
        if args.table:
            folder_total = print_table(folder_counts, folder_type)
        else:
            print(f"\n{folder_type.capitalize()}:")
            folder_total = 0
            for split_type in ['train', 'test']:
                folder_total += print_counts(folder_counts, split_type)
            print(f"  Total: {folder_total}")
        
        grand_total += folder_total
    
    print("\n" + "=" * 80)
    print(f"Grand Total: {grand_total}")
    print("=" * 80)

if __name__ == "__main__":
    main() 