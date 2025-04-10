# Enhancing OCR Accuracy Through CNN-Based Deobfuscation of Adversarial Text

## Overview

This is an MSc Computer Science project that implements a CNN-based deobfuscation model for letter classification (which can be extended to word classification in the future). The system is designed to counter adversarial text obfuscation techniques, where characters are deliberately transformed to confuse automated recognition while preserving human readability.

Common obfuscation techniques include:

- Replacing letters with look-alike characters (e.g., '@' for 'a', '0' for 'o')
- Using characters from other scripts or alphabets
- Using symbols that visually resemble standard characters
- Inserting zero-width spaces or other invisible characters

### The Project at a Glance

This project focuses on improving Optical Character Recognition (OCR) accuracy by tackling the challenge of adversarial text obfuscation. Here's a comprehensive overview:

**Core Problem:**
The project addresses a significant challenge in text recognition systems: deliberate text obfuscation. This is when text is intentionally modified to fool automated systems while remaining readable to humans, using techniques like character substitution, alternative scripts, and invisible characters.

**Proposed Solution:**
The project implements a novel two-stage approach:

1. A CNN-based deobfuscation system as a preprocessing step
2. Traditional OCR (Tesseract) for final text recognition

The key innovation is using deep learning (specifically CNNs) to "normalize" obfuscated text before passing it to OCR, rather than trying to modify the OCR system itself.

**Project Implementation:**
The project is being developed in phases:

- **Phase 1 (Current)**: Focusing on single-character deobfuscation
- **Phase 2 (Planned)**: Extending to word-level deobfuscation

**Dataset:**
The project uses a comprehensive dataset consisting of:

- Regular characters: ~19,000 images
- Obfuscated characters: ~19,400 images
- Split 80/20 for training/testing
- Uses multiple fonts (164 regular fonts, 73 obfuscated fonts)
- Includes both computer-generated and handwritten samples

**Technical Architecture:**

- Uses PyTorch for deep learning implementation
- Implements a CNN-based architecture for deobfuscation
- Includes comprehensive training, evaluation, and inference pipelines
- Has robust data preprocessing and augmentation
- Includes detailed evaluation metrics (pixel-level, OCR-specific, visual comparisons)

**Academic Foundation:**
The project builds on recent research in adversarial text and OCR robustness, citing works from:

- Akhtar et al. (2022) on adversarial text obfuscation attacks
- Song and Shmatikov (2018) on fooling OCR systems
- Imam et al. (2022) on enhancing OCR robustness

**Practical Applications:**
This research has potentially significant real-world applications in:

- Content moderation systems
- Security measures
- Automated data extraction
- Anti-spam systems
- Digital forensics

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
├── configs/          # Configuration files for models and training
├── data/            # Training, validation, and test datasets (untracked)
├── fonts/           # Font lists for regular and obfuscated character generation
├── models/          # Saved model checkpoints
├── notebooks/       # Jupyter notebooks for exploration and visualization
├── outputs/         # Training outputs, checkpoints, and results
├── results/         # Evaluation results and comparisons
├── scripts/         # Utility scripts for training, evaluation, and inference
│   ├── process_files.py          # Cleans up and renames files in directories
│   ├── count_character_files.py  # Counts and reports file statistics
│   ├── split_dataset.py          # Splits a dataset into train/test sets
│   ├── remove_fonts.py           # Removes images with specific fonts from the dataset
│   └── extract_fonts.py          # Extracts font information from image filenames
├── src/             # Source code
│   ├── data/        # Data loading and preprocessing modules
│   ├── models/      # CNN architecture definitions
│   ├── train/       # Training pipeline
│   ├── evaluate/    # Evaluation metrics and pipeline
│   └── utils/       # Utility functions
├── tests/           # Unit tests for project components
├── requirements.txt     # Python package dependencies
├── .gitignore          # Git ignore rules
├── .gitattributes      # Git attributes configuration
└── LICENSE.md          # Project license information
└── README.md          # Project information
```

## Getting Started

### Prerequisites

1. Python 3.9
2. PyTorch 2.0+
3. Required Python packages (see requirements.txt):
   - torch
   - torchvision
   - numpy
   - matplotlib
   - scikit-learn
   - tqdm
   - pyyaml
4. Download the regular character dataset from the link provided above
5. Download all the fonts in the fonts/fonts_obfuscated.txt file on your computer

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

We are in **Phase 1** of the project, concerned with character image recognition

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
│   └── obfuscated/   # Obfuscated character images
│       ├── train/    # Training set (~80%)
│       └── test/     # Test set (~20%)

fonts/
├── fonts_regular.txt    # List of fonts used for regular characters
└── fonts_obfuscated.txt # List of fonts used for obfuscated characters
```

**Note**: The `words/` directory is reserved for Phase 2 of the project and is not currently used.

#### Dataset Generation

##### Regular characters

The regular character images should be downloaded from the source provided. After downloading, put all 26 character set (a-z, each of which is a folder containing png images of letters), into the `data/characters/regular/train` directory. Then, run this command to create a train-test split of the regular characters:

```bash
python3.9 scripts/split_dataset.py
```

##### Obfuscated characters

To generate an initial dataset of obfuscated character images, run:

```bash
# Generate obfuscated character images in the `data/characters/obfuscated/train` directory
python3.9 src/data/gen_obf_from_fontlist.py

# Create a train-test split of the obfuscated character images
python3.9 scripts/split_dataset.py --data_dir data/characters/obfuscated
```

(Optional) If you wish to generate more obfuscated images to bolster the dataset, including blurred, rotated character images, run the following command (NOTE: this could take several hours). The train-test split will be automatically created by the script.

```bash
python3.9 src/data/gen_obf_from_regular.py
```

After generating the dataset images, you should clean up the character dataset directories by removing any non-PNG files and renaming the image files to a consistent numerical format. You can do this by running:

```bash
python3.9 scripts/process_files.py --directory ./data/characters --recursive
```

This will:

1. Remove all non-PNG files (including .DS_Store files, on MacOS)
2. Rename all PNG files to a three-digit numerical format (001.png, 002.png, etc.)
3. Process all subdirectories in the same manner, recursively

##### Useful Scripts

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

### Model Architecture

The project uses a deep Convolutional Neural Network (CNN) specifically designed for character deobfuscation. The architecture balances complexity with efficiency, using approximately 2.5M parameters.

#### Network Structure

**Input Layer**

- Accepts 28x28x1 grayscale images
- Normalized pixel values in range [0, 1]

**Feature Extraction Blocks**

1. **First Convolutional Block** (Input → 32 channels)

   - Two Conv2D layers (3x3 kernel, padding=1)
   - Batch Normalization after each conv
   - ReLU activation
   - MaxPool2D (2x2, stride=2)
   - Light Dropout (rate=0.25)
   - Output: 14x14x32

2. **Second Convolutional Block** (32 → 64 channels)

   - Two Conv2D layers (3x3 kernel, padding=1)
   - Batch Normalization after each conv
   - ReLU activation
   - MaxPool2D (2x2, stride=2)
   - Light Dropout (rate=0.25)
   - Output: 7x7x64

3. **Third Convolutional Block** (64 → 128 channels)
   - Two Conv2D layers (3x3 kernel, padding=1)
   - Batch Normalization after each conv
   - ReLU activation
   - MaxPool2D (2x2, stride=2)
   - Standard Dropout (rate=0.5)
   - Output: 3x3x128

**Classification Layers**

1. **Flatten Layer**

   - Converts 3x3x128 feature maps to 1152-dimensional vector

2. **First Dense Block** (1152 → 512)

   - Linear transformation
   - Batch Normalization
   - ReLU activation
   - Dropout (rate=0.5)

3. **Second Dense Block** (512 → 256)

   - Linear transformation
   - Batch Normalization
   - ReLU activation
   - Dropout (rate=0.5)

4. **Output Layer** (256 → 26)
   - Linear transformation to class logits
   - Output dimension: 26 (one per character)

**Note**: The architecture is implemented in `src/models/letter_classifier.py` and can be configured through the YAML configuration file.

#### Key Design Features

1. **Progressive Feature Extraction**

   - Gradually increases feature complexity (32 → 64 → 128 channels)
   - Systematic spatial reduction (28x28 → 14x14 → 7x7 → 3x3)
   - Maintains spatial information through padding

2. **Regularization Strategy**

   - Batch Normalization for stable training
   - Progressive Dropout:
     - Light in early layers (0.25)
     - Heavier in later layers (0.5)
   - Weight initialization using He normal distribution

3. **Architectural Choices**
   - Double convolution blocks inspired by VGG
   - Skip connections within blocks for better gradient flow
   - Balanced depth vs width for efficient learning

#### Implementation Details

```python
# Feature extraction example (first block)
nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout2d(p=0.25)
)

# Classification example (first dense block)
nn.Sequential(
    nn.Linear(1152, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5)
)
```

#### Training Considerations

- **Memory Efficiency**: ~2.5M parameters, suitable for GPU training
- **Batch Size**: Recommended 32-128 depending on available memory
- **Gradient Flow**: Good gradient propagation due to:
  - Proper initialization
  - Batch normalization
  - Residual connections
- **Regularization Balance**: Combined use of:
  - Architectural regularization (depth)
  - Explicit regularization (dropout)
  - Implicit regularization (batch norm)

### Training a Model

#### Training on Google Colab

Important Note: It is **highly** recommended to run the training notebook (`notebooks/colab_train_model.ipynb`) on Google Colab, rather than try to run the training script locally. This is due to the large size of the dataset and the computational resources required to train the model. You can download the entire training and testing dataset from Google Drive (link provided in the notebook) and use it for training on Colab.

#### Training Locally

To train a model, you'll need to:

1. First ensure your data is properly organized in the `data/characters` directory:

```
data/characters/
├── regular/
│   ├── train/
│   └── test/
└── obfuscated/
    ├── train/
    └── test/
```

2. Run the training script:

```bash
python3.9 src/train/train.py
```

The training process will:

1. **Data Loading**:

   - Load both regular and obfuscated character datasets
   - Apply data augmentation for training data (random rotation, affine transformations)
   - Normalize images to [0, 1] range
   - Convert images to grayscale

2. **Model Training**:

   - Train the model for the specified number of epochs
   - Use Adam optimizer with configurable learning rate
   - Apply learning rate scheduling (ReduceLROnPlateau)
   - Implement early stopping to prevent overfitting

3. **Monitoring and Visualization**:

   - Display real-time progress bars with loss and accuracy
   - Log training metrics after each epoch
   - Generate training history plots (loss and accuracy curves)
   - Create confusion matrices every 10 epochs
   - Calculate and log per-character accuracy

4. **Checkpointing**:
   - Save model checkpoints every 5 epochs
   - Keep track of the best model based on validation loss
   - Save training history and optimizer state

The training outputs will be saved in the specified output directory:

```
outputs/letter_classifier/
├── best_model.pth           # Best model based on validation loss
├── checkpoint_epoch_*.pth   # Regular checkpoints
├── training_history.png     # Training curves
└── confusion_matrix.png     # Final confusion matrix
```

Training progress will be displayed in real-time:

```
Loading datasets...
- Regular dataset: 18,981 images
- Obfuscated dataset: 19,385 images
- Combined dataset: 38,366 images
- Training set: 30,693 images (80%)
- Validation set: 7,673 images (20%)

Initializing model...
- Input shape: [28, 28, 1]
- Number of classes: 26 (a-z)
- Model architecture: LetterClassifierCNN

Starting training...
Epoch 1/100
[████████████████████████████████████████████████████] 100%
Train Loss: 3.2589, Train Acc: 12.45%
Val Loss: 3.1423, Val Acc: 14.32%
Learning Rate: 0.001000

Epoch 2/100
[████████████████████████████████████████████████████] 100%
Train Loss: 2.9876, Train Acc: 18.76%
Val Loss: 2.8765, Val Acc: 20.12%
Learning Rate: 0.001000
...
```

The training configuration can be modified in `configs/train_config.yaml`. Key settings include:

- Dataset paths and image size
- Training parameters (epochs, batch size, learning rate)
- Learning rate scheduler settings
- Early stopping configuration
- Data augmentation parameters

Training will automatically stop if:

- The specified number of epochs is reached
- Early stopping is triggered (no improvement for N epochs)
- The minimum learning rate is reached

After training completes, you'll find:

- The best model saved as `best_model.pth`
- Training history plots showing loss and accuracy curves
- A final confusion matrix with per-character accuracy
- Regular checkpoints for potential training resumption

### Evaluating a Model

To evaluate a trained model:

```bash
python3.9 src/evaluate/evaluate.py --model_path outputs/letter_classifier/best_model.pth --config configs/train_config.yaml --data_dir data/characters
```

Options:

- `--model_path`: Path to the trained model checkpoint
- `--config`: Path to the model config file
- `--data_dir`: Path to the dataset directory
- `--output_dir`: Path to the output directory
- `--batch_size`: Batch size for evaluation (default: 32)
- `--gpu`: Use GPU for evaluation

The evaluation script will:

1. Load the trained model
2. Evaluate on the test dataset
3. Generate and save:
   - Confusion matrix
   - Classification report
   - Evaluation metrics (loss, accuracy)
   - Detailed results in JSON format

### Running Inference

To run inference on new images:

```bash
python3.9 src/inference/inference.py --model_path outputs/letter_classifier/best_model.pth --config configs/train_config.yaml --input path/to/image.png
```

Options:

- `--model_path`: Path to the trained model checkpoint
- `--config`: Path to the model config file
- `--input`: Path to input image or directory of images
- `--output_dir`: Path to output directory (default: outputs/inference)
- `--gpu`: Use GPU for inference

The inference script will:

1. Process single images or entire directories
2. Generate predictions for each image
3. Save results including:
   - Input image with prediction overlay
   - Confidence scores
   - Detailed results in JSON format
   - Visualizations in a dedicated directory

### Testing

The project includes comprehensive unit tests for all major components. The tests are organized in the `tests/` directory and use a centralized test configuration.

#### Test Structure

```
tests/
├── test_config.yaml        # Centralized test configuration
├── test_utils.py          # Common test utilities
├── test_data_loader.py    # Data loader tests
├── test_model.py          # Model architecture tests
├── test_trainer.py        # Training pipeline tests
├── test_evaluation.py     # Evaluation metrics tests
└── test_inference.py      # Inference functionality tests
```

#### Test Configuration

The test configuration file (`test_config.yaml`) contains settings for:

- Model parameters (input channels, number of classes, dropout rates)
- Data loader settings (batch size, augmentation, test characters)
- Training parameters (epochs, learning rate, early stopping)
- Evaluation metrics and thresholds
- Inference settings
- Environment configuration (GPU usage, random seeds)
- Test data generation parameters

#### Running Tests

1. Run all tests:

```bash
python -m unittest discover tests/
```

2. Run specific test file:

```bash
python -m unittest tests/test_model.py
```

3. Run with coverage report:

```bash
coverage run -m unittest discover tests/
coverage report
```

#### Test Categories

1. **Data Loader Tests**

   - Dataset initialization
   - Data augmentation
   - Character mapping
   - Batch loading
   - Error handling

2. **Model Tests**

   - Architecture initialization
   - Forward pass
   - Gradient flow
   - Parameter initialization
   - Dropout behavior

3. **Trainer Tests**

   - Training loop
   - Validation
   - Checkpointing
   - Early stopping
   - Learning rate scheduling

4. **Evaluation Tests**

   - Confusion matrix generation
   - Accuracy calculation
   - Per-class metrics
   - Result saving
   - Visualization

5. **Inference Tests**
   - Image preprocessing
   - Single/batch prediction
   - Confidence thresholds
   - Error handling
   - Visualization generation

#### Test Utilities

The `test_utils.py` module provides common utilities for tests:

- `load_test_config()`: Load test configuration
- `setup_test_environment()`: Set up test environment with random seeds
- `get_test_model()`: Create test model instance
- `create_test_image()`: Generate test images
- `get_test_batch()`: Create test batches

#### Writing New Tests

When adding new tests:

1. Use the test configuration for parameters
2. Follow the existing test structure
3. Include proper setup and teardown
4. Test both success and error cases
5. Add docstrings for test methods
6. Use the test utilities where appropriate

Example:

```python
def test_new_feature(self):
    """Test new feature functionality."""
    # Use configuration
    param = self.config['feature']['param']

    # Use test utilities
    test_data = create_test_image(self.config)

    # Test the feature
    result = self.feature(test_data, param)

    # Assert expected behavior
    self.assertIsNotNone(result)
    self.assertEqual(result.shape, expected_shape)
```

## Configuration

The model configuration is specified in YAML files in the `configs` directory. The main configuration file is `train_config.yaml`, which contains settings for:

- **Model**: Architecture parameters, input shape, number of classes
- **Training**: Batch size, epochs, learning rate, optimizer settings
- **Data**: Dataset paths, image size, augmentation settings
- **Output**: Directory settings, checkpoint frequency

Example:

```yaml
model:
  architecture: "LetterClassifierCNN"
  input_shape: [28, 28, 1] # Height, Width, Channels
  num_classes: 26 # a-z
  dropout_rate: 0.5

training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "adam"
  scheduler:
    use: true
    type: "reduce_on_plateau"
    patience: 5
    factor: 0.5
    min_lr: 0.00001
  early_stopping:
    use: true
    patience: 15
    min_delta: 0.001

data:
  regular:
    train_dir: "data/characters/regular/train"
    test_dir: "data/characters/regular/test"
  obfuscated:
    train_dir: "data/characters/obfuscated/train"
    test_dir: "data/characters/obfuscated/test"
  image_size: [28, 28]
  validation_split: 0.2
  shuffle: true
  augmentation:
    use: true
    rotation_range: 10
    zoom_range: 0.1
    width_shift_range: 0.1
    height_shift_range: 0.1
    brightness_range: [0.8, 1.2]
    random_noise: 0.01

output:
  dir: "outputs/letter_classifier"
  save_frequency: 5
  keep_best: true
```

## Evaluation

The system is evaluated using multiple metrics and approaches:

1. **Model Performance Metrics**:

   - Classification accuracy (overall and per-character)
   - Loss curves (training and validation)
   - Confusion matrix
   - Learning rate progression

2. **OCR Pipeline Comparison**:

   - Direct OCR on obfuscated text
   - OCR after CNN deobfuscation
   - Character-level accuracy comparison
   - Word-level accuracy comparison (Phase 2)

3. **Visual Analysis**:
   - Sample predictions
   - Confusion matrices
   - Training history plots
   - Learning rate schedules

Evaluation results are saved in the specified output directory:

```
outputs/letter_classifier/
├── best_model.pth           # Best model based on validation loss
├── checkpoint_epoch_*.pth   # Regular checkpoints
├── training_history.png     # Training curves
├── confusion_matrix.png     # Final confusion matrix
└── evaluation_results.json  # Detailed evaluation metrics
```

**Note**: The OCR pipeline comparison is currently implemented for character-level evaluation only. Word-level evaluation will be added in Phase 2.

## Academic References

This project builds on research in adversarial text obfuscation and OCR robustness:

- Akhtar et al. (2022). "Adversarial text: Understanding and mitigating text obfuscation attacks"
- Song and Shmatikov (2018). "Fooling OCR systems with adversarial text images"
- Imam et al. (2022). "Enhancing OCR robustness against deliberate text manipulations"

## License

N/A

## Contributors

- [Ashraf M. Yusuf](https://github.com/cs-amy)
