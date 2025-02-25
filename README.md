# MSc Project 

### Topic
Comparative Evaluation of End-to-End CNN Models Versus Transformer-Based OCR Pipelines (Google Vision OCR and Microsoft TrOCR Combined with Detoxify and HateBERT) for Detecting Obfuscated Profanity in Digital Content

### Description
This project is an MSc-level research endeavor conducted at University of Hertfordshire under the supervision of Dr Jan T. Kim. The aim is to compare a direct CNN-based approach with multiple state-of-the-art OCR pipelines for detecting obfuscated profanity in digital content. The project involves:
- Generating a diverse dataset of images containing profanity in plain, obfuscated, and multiple case variants.
- Building an end-to-end CNN model that directly detects obfuscated profanity from images.
- Implementing two OCR pipelines using Google Vision OCR and Microsoft TrOCR, each coupled with two BERT-based text classifiers (Detoxify and HateBERT).
- Conducting a rigorous comparative evaluation in terms of accuracy, robustness, computational efficiency, and interpretability.

### Project Structure
See directory-structure.md in the project root

### Installation
##### Clone the Repository:
```
git clone https://github.com/cs-amy/project-codebase.git
cd project-codebase
```

##### Create a Virtual Environment (using venv):
```
python3 -m venv venv
```

##### Activate the environment:
- On Windows:
```
venv\Scripts\activate
```

- On macOS/Linux:
```
source venv/bin/activate
```

##### Install Dependencies:
```
pip3 install -r requirements.txt
```

### Usage
The following steps are included in the project:
- Image Generation
- Model Training & Evaluation

##### Image Generation
- Input Files:
    - data/profanities.txt: contains 40 profanities.
- Output:
    - Images are generated in the data/raw directory (not included in the repository due to its size). With multi-case enabled and including obfuscations, each profanity word produces 40 images (white and black backgrounds), for a total of 1600 images.
    - Sample images are provided in the data/raw_sample directory.

##### Model Training & Evaluation
As the project evolves, the next steps will include:
- Pre-processing the generated images (resizing, normalization, etc.).
- Training the CNN model (proposed file: src/cnn_model.py).
- Running OCR pipelines with Google Vision OCR and Microsoft TrOCR, coupled with Detoxify and HateBERT (proposed files: src/ocr_pipeline.py and src/text_filter.py).
- Conducting comparative evaluations (proposed file: src/evaluation.py).

These steps can be executed via the main entry point:
```
python3 -m src.main
```

### Testing
Before running unit tests: 
- Ensure that the virtual environment is activated.
- Install pytest if not already installed:
```
pip3 install pytest
```

Run unit tests with:
```
pytest
```
or
```
python3 -m unittest discover
```

### Future Work
Upcoming steps include:
- Expanding and refining the dataset.
- Pre-processing images and preparing data for model training.
- Developing and fine-tuning the CNN model.
- Integrating and evaluating OCR pipelines with text classifiers.
- Comprehensive evaluation, error analysis, and report finalization.

### Notes
- This README will be updated continuously as the project progresses.
- You may encounter bugs (this is still a work in progress). Please report them to me. Thank you!
