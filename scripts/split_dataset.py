"""
Script to create a train/test split for the character dataset.
Moves a portion of images from train to test directory while maintaining
the same directory structure.
"""

import os
import random
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Create train/test split for character dataset.')
    parser.add_argument('--data_dir', type=str, default='data/characters/regular', 
                        help='Directory containing the character dataset')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Ratio of data to move to test set (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    
    # Ensure test directory exists
    os.makedirs(test_dir, exist_ok=True)
    
    # Get list of character subdirectories
    char_dirs = [d for d in os.listdir(train_dir) 
                if os.path.isdir(os.path.join(train_dir, d))]
    
    total_files = 0
    moved_files = 0
    
    # Process each character directory
    for char in char_dirs:
        # Create corresponding directory in test if it doesn't exist
        char_test_dir = os.path.join(test_dir, char)
        os.makedirs(char_test_dir, exist_ok=True)
        
        # Get all PNG files in the character's train directory
        char_train_dir = os.path.join(train_dir, char)
        files = [f for f in os.listdir(char_train_dir) 
                if f.endswith('.png') and os.path.isfile(os.path.join(char_train_dir, f))]
        
        total_files += len(files)
        
        # Determine number of files to move
        num_test = int(len(files) * args.test_ratio)
        
        # Randomly select files to move
        test_files = random.sample(files, num_test)
        moved_files += len(test_files)
        
        # Move files to test directory
        for file in test_files:
            src = os.path.join(char_train_dir, file)
            dst = os.path.join(char_test_dir, file)
            shutil.move(src, dst)
            
        print(f"Character '{char}': Moved {len(test_files)} of {len(files)} files to test")
    
    print(f"\nDone! Moved {moved_files} of {total_files} files to test set ({moved_files/total_files:.1%})")

if __name__ == "__main__":
    main()
