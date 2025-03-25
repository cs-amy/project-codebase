# Enhancing OCR Accuracy Through CNN-Based Deobfuscation of Adversarial Text

## Overview
This project implements a CNN-based deobfuscation system as a preprocessing step for Optical Character Recognition (OCR). The system is designed to counter adversarial text obfuscation techniques, where characters are deliberately transformed to confuse automated recognition while preserving human readability.

Common obfuscation techniques include:
- Replacing letters with look-alike characters (e.g., '@' for 'a', '0' for 'o')
- Using characters from other scripts or alphabets
- Using symbols that visually resemble standard characters
- Inserting zero-width spaces or other invisible characters

## Problem Statement
OCR systems face challenges from adversarial text obfuscation—deliberate transformations that confuse automated recognition while preserving human readability. These manipulations hinder accurate text extraction, reducing the effectiveness of security measures, content moderation, and automated data extraction systems.

## Approach
Our approach employs a Convolutional Neural Network (CNN) as a preprocessing step before applying traditional OCR (Tesseract). The deobfuscation CNN converts obfuscated text images to their standard forms, enabling more accurate recognition.

The project provides a complete pipeline for:
1. Generating datasets of obfuscated and standard character images
2. Training CNN models to convert obfuscated images to standard form
3. Evaluating model performance
4. Running inference on new images

The implementation is divided into two stages:
1. **Single-letter deobfuscation**: Converting individual obfuscated characters to their standard forms
2. **Word-level deobfuscation**: Extending the approach to handle complete words

## Project Structure
```
project-codebase/
├── .venv/          # Virtual ensironment (untracked)
├── configs/        # Configuration files for models and training
├── data/           # Training, validation, and test datasets (untracked)
├── fonts/          # Font lists for regular and obfuscated character generation
├── models/         # Saved model checkpoints
├── notebooks/      # Jupyter notebooks for exploration and visualization
├── results/        # Evaluation results and comparisons
├── scripts/        # Utility scripts for training, evaluation, and inference
│   ├── process_files.py          # Cleans up and renames files in directories
│   ├── count_character_files.py  # Counts and reports file statistics
│   ├── split_dataset.py          # Splits a dataset into train/test sets
│   ├── remove_fonts.py           # Removes images with specific fonts from the dataset
│   └── extract_fonts.py          # Extracts font information from image filenames
├── src/            # Source code
│   ├── data/       # Data loading and preprocessing modules
│   ├── models/     # CNN architecture definitions
│   ├── train/      # Training pipeline
│   ├── evaluate/   # Evaluation metrics and pipeline
│   └── utils/      # Utility functions
├── tests/          # Unit tests for project components
├── requirements.txt    # Python package dependencies
├── .gitignore         # Git ignore rules
├── .gitattributes     # Git attributes configuration
└── LICENSE.md         # Project license information
```

## Key Components

### Models


### Training

- **Trainer**: Handles the training loop, validation, checkpointing, and early stopping
- **Loss Functions**: Various loss functions including BCE, Dice, SSIM, and combinations

### Evaluation

- **Evaluator**: Computes metrics on test set and creates visualizations
- **Pipeline Comparison**: Compares OCR accuracy with and without deobfuscation
- **Confusion Matrix**: Visualizes OCR errors for different characters

### Inference

- **Inference Script**: Processes new images with a trained model
- **Visualization**: Creates side-by-side comparisons of input and output

### Scripts

The project includes several utility scripts for data preparation and model operation:

- **process_files.py**: A utility script for cleaning up and organizing image directories. It removes non-PNG files (including .DS_Store files) and renames PNG files to a consistent three-digit numerical format (e.g., 001.png, 002.png). Can process directories recursively.
  ```bash
  # Process a single directory
  python3.9 scripts/process_files.py --directory ./data/characters/regular
  
  # Process directory recursively
  python3.9 scripts/process_files.py --directory ./data/characters --recursive
  
  # Process with sorting enabled
  python3.9 scripts/process_files.py --directory ./data/characters --recursive --sort
  ```

- **count_character_files.py**: A script that analyzes and reports statistics about character image files in the dataset. It counts files in each character directory (a-z) for both regular and obfuscated datasets, and can display results in either a detailed format or a table format.
  ```bash
  # Display in default format
  python3.9 scripts/count_character_files.py
  
  # Display in table format
  python3.9 scripts/count_character_files.py --table
  ```

- **split_dataset.py**: A utility for creating a train/test split for the character dataset. It moves a portion of images from train to test directory while maintaining the same directory structure. The script processes each character directory separately to maintain balanced splits.
  ```bash
  # Split with default 80/20 ratio
  python3.9 scripts/split_dataset.py --data_dir data/characters/regular
  
  # Split with custom test ratio
  python3.9 scripts/split_dataset.py --data_dir data/characters/regular --test_ratio 0.25
  
  # Split with specific random seed
  python3.9 scripts/split_dataset.py --data_dir data/characters/regular --seed 42
  ```

- **remove_fonts.py**: A script for removing obfuscated character images that use specific fonts. It searches through the obfuscated dataset and deletes images that contain any of the specified font names in their filenames. The script processes both train and test directories.
  ```bash
  # Remove default list of fonts
  python3.9 scripts/remove_fonts.py
  
  # Remove specific fonts
  python3.9 scripts/remove_fonts.py --fonts_to_remove "Arial" "Times New Roman"
  ```

- **extract_fonts.py**: A utility that extracts and compiles a list of all fonts used in the regular character dataset. It parses the filenames of images, extracts font information, and saves a list of unique fonts to a file.
  ```bash
  # Extract fonts from default directory
  python3.9 scripts/extract_fonts.py
  
  # Extract fonts from specific directory
  python3.9 scripts/extract_fonts.py --data_dir data/characters/regular/train
  
  # Extract fonts and save to specific file
  python3.9 scripts/extract_fonts.py --output_file fonts_regular.txt
  ```

## Getting Started

### Prerequisites

1. Python 3.9
2. PyTorch 2.0+
3. TensorFlow 2.0+ (optional)
4. Tesseract OCR engine (system dependency)
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`
   - Windows: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
5. Download the regular character dataset from the link provided above
6. Download all the fonts in the fonts/fonts_obfuscated.txt file on your computer

### Installation

1. Clone the repository:
   ```bash
   # Clone the repository
   git clone https://github.com/cs-amy/project-codebase.git
   cd project-codebase
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python3.9 -m venv .venv
   source .venv/bin/activate 
   # On Windows, use: .venv/Scripts/activate (CMD) or .venv/Scripts/Activate.ps1 (Powershell)
   ```
   You may find this guide useful: [How to set up virtual environments](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/)

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Character Dataset

The regular character dataset used in this project can be obtained from the following source:

The original dataset can be found in the Mondragon Unibertsitatea Repository: [https://gitlab.danz.eus/datasharing/ski4spam](https://gitlab.danz.eus/datasharing/ski4spam)

The training dataset consists of:
- Alphabetic letters (a-z) written using different fonts and styles (regular, cursive, bold, cursive+bold)
- Handwritten letters: English handwriting from the Chars74k dataset [2] which is available at http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/.

**Note**: The use of the Chars74k dataset (or parts of it) implies acknowledging its source and citing the relevant paper [2] in related publications.

The regular image datasets have been conceived and developed by The Data Analysis and Cybersecurity research team of Mondragon Unibertsitatea: [https://www.mondragon.edu/en/research-transfer/engineering-technology/research-and-transfer-groups/-/mu-inv-mapping/grupo/analisis-de-datos-y-ciberseguridad](https://www.mondragon.edu/en/research-transfer/engineering-technology/research-and-transfer-groups/-/mu-inv-mapping/grupo/analisis-de-datos-y-ciberseguridad)

[1] Flamand, Eveline, and Anne-Marie Simon-Vandenbergen. Deciphering L33t5p34k: Internet Slang On Message Boards. 2008. Available at https://lib.ugent.be/catalog/rug01:001414289

[2] T. E. de Campos, B. R. Babu and M. Varma. Character recognition in natural images. In Proceedings of the International Conference on Computer Vision Theory and Applications (VISAPP), Lisbon, Portugal, February 2009.

#### Dataset Statistics

Our current dataset contains:
- **Total Images:** 38,366 PNG files
- **Regular Character Images:** 18,981 images (49.4% of total)
- **Obfuscated Character Images:** 19,385 images (50.6% of total)

#### Train/Test Splits
- **Regular Characters:**
  - Training: 15,190 images (80%)
  - Test: 3,791 images (20%)
  - Train/Test Ratio: 4.00:1

- **Obfuscated Characters:**
  - Training: 15,510 images (80%)
  - Test: 3,875 images (20%)
  - Train/Test Ratio: 4.00:1

#### Character Distribution
- **Regular Training Set:** 500-700 images per character
- **Obfuscated Training Set:** 500-700 images per character

#### Font Information
The dataset uses a variety of fonts:
- **Regular Character Fonts:** 164 distinct fonts
- **Obfuscated Character Fonts:** 73 distinct fonts

#### Dataset Organization

The dataset is organized as follows:
```
data/
├── characters/
│   ├── regular/      # Standard character images
│   │   ├── train/    # Training set (~80%)
│   │   └── test/     # Test set (~20%)
│   ├── obfuscated/   # Obfuscated character images
│   │   ├── train/    # Training set (~80%)
│   │   └── test/
│   └── obfuscated_backup/  # Backup of obfuscated data before cleaning
└── words/            # Word-level examples

fonts/
├── fonts_regular.txt    # List of fonts used for regular characters
└── fonts_obfuscated.txt # List of fonts used for obfuscated characters
```

#### Dataset Generation

##### Regular characters

The regular character images should be downloaded from the source provided. After downloading the character set (a-z), 

##### Obfuscated characters

To generate a dataset of obfuscated character images, run:

```bash
python3.9 src/data/gen_obf_from_fontlist.py
# This will generate the 
```

```bash
python3.9 scripts/generate_dataset.py --output_dir data/characters --num_samples 10000
```

Options:
- `--output_dir`: Path to output directory
- `--num_samples`: Number of samples to generate
- `--image_size`: Size of the output images
- `--train_ratio`: Ratio of training samples
- `--val_ratio`: Ratio of validation samples
- `--test_ratio`: Ratio of test samples

After generating the dataset images, you should clean up the directories by removing any non-PNG files and renaming the PNG files to a consistent numerical format. You can do this by running:

```bash
python3.9 scripts/process_files.py --directory ./data/characters --recursive
```

This will:
1. Remove all non-PNG files (including .DS_Store files in MacOS)
2. Rename all PNG files to a three-digit numerical format (001.png, 002.png, etc.)
3. Process all subdirectories recursively

### Training a Model

To train a model:

```bash
python scripts/train_model.py --config configs/model_config.yaml --data_dir data/characters --output_dir results
```

Options:
- `--config`: Path to the config file
- `--data_dir`: Path to the dataset directory
- `--output_dir`: Path to the output directory
- `--model_name`: Model architecture name (overrides config)
- `--batch_size`: Batch size (overrides config)
- `--epochs`: Number of epochs (overrides config)
- `--learning_rate`: Learning rate (overrides config)
- `--resume`: Path to checkpoint to resume training from
- `--gpu`: Use GPU for training
- `--seed`: Random seed

### Evaluating a Model

To evaluate a trained model:

```bash
python scripts/evaluate_model.py --model_path results/run_TIMESTAMP/model_final.pth --config results/run_TIMESTAMP/config.yaml --data_dir data/characters
```

Options:
- `--model_path`: Path to the trained model checkpoint
- `--config`: Path to the model config file
- `--data_dir`: Path to the dataset directory
- `--output_dir`: Path to the output directory
- `--batch_size`: Batch size for evaluation
- `--threshold`: Threshold for binarizing predictions
- `--num_samples`: Number of sample visualizations to save
- `--pipeline_comparison`: Perform OCR pipeline comparison
- `--confusion_matrix`: Generate confusion matrix for OCR results
- `--gpu`: Use GPU for evaluation

### Running Inference

To run inference on new images:

```bash
python scripts/inference.py --model_path results/run_TIMESTAMP/model_final.pth --config results/run_TIMESTAMP/config.yaml --input path/to/image.png
```

Options:
- `--model_path`: Path to the trained model checkpoint
- `--config`: Path to the model config file
- `--input`: Path to input image or directory of images
- `--output_dir`: Path to output directory
- `--threshold`: Threshold for binarizing predictions
- `--word_mode`: Process inputs as words instead of single characters
- `--perform_ocr`: Perform OCR on the input and output images
- `--gpu`: Use GPU for inference

### Testing


## Configuration

The model configuration is specified in YAML files in the `configs` directory. The main configuration file is `model_config.yaml`, which contains settings for:

- **Model Architecture**: Type, input shape, filters, dropout, etc.
- **Training**: Batch size, epochs, learning rate, optimizer, etc.
- **Data**: Train/val/test split, augmentation, etc.
- **Inference**: Confidence threshold, post-processing, etc.

Example:

```yaml
model:
  name: "OCRDeobfuscatorCNN"
  architecture: "unet"  # Options: "unet", "resnet", "autoencoder"
  input_shape: [64, 64, 1]  # Height, Width, Channels
  filters:
    - 64
    - 128
    - 256
    - 512
  dropout_rate: 0.2
  use_batch_norm: true
  activation: "relu"
  final_activation: "sigmoid"

training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"  # Options: "adam", "sgd", "rmsprop"
  # ... more settings ...
```

Character mappings for obfuscation are defined in `configs/character_mapping.yaml`. This file maps standard characters to their obfuscated variants, allowing for customization of the obfuscation patterns.

## Evaluation

The system is evaluated by comparing two pipelines:
1. Obfuscated text → Tesseract OCR
2. Obfuscated text → CNN deobfuscator → Tesseract OCR

Performance metrics include:
- Pixel-level metrics: accuracy, Dice coefficient, IoU, SSIM
- OCR-specific metrics: character/word accuracy, string edit distance
- Visual comparisons: side-by-side visualizations of original and deobfuscated text

After training, results are saved in the specified output directory, including:
- **Model Checkpoints**: Saved at regular intervals and at the end of training
- **Training History**: Loss curves and learning rate
- **Sample Visualizations**: Input, output, and target images
- **Evaluation Metrics**: Detailed performance measurements
- **Pipeline Comparison**: OCR accuracy with and without deobfuscation
- **Confusion Matrices**: Visualization of OCR errors

## Academic References

This project builds on research in adversarial text obfuscation and OCR robustness:

- Akhtar et al. (2022). "Adversarial text: Understanding and mitigating text obfuscation attacks"
- Song and Shmatikov (2018). "Fooling OCR systems with adversarial text images"
- Imam et al. (2022). "Enhancing OCR robustness against deliberate text manipulations"

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributors

- [Ashraf M. Yusuf](https://github.com/cs-amy)