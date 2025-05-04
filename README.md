# Enhancing OCR Accuracy Through CNN-Based De-Obfuscation of Adversarial Text

**_A Master’s in Computer Science Project — University of Hertfordshire, MSc Computer Science (7COM1040-0206‑2024)_**

Student: _Ashraf Muhammed Yusuf_

## Abstract

Adversarial text obfuscation—the deliberate
replacement of letters with visually similar symbols, homoglyphs, or zero-width
characters—has emerged as a low-effort yet highly effective method for evading
automated content filters that rely on optical character recognition (OCR).
Off-the-shelf systems such as Tesseract routinely misrecognise or ignore such
perturbations, allowing toxic or malicious content embedded in images to slip
through moderation pipelines. This project investigates whether a lightweight,
vision-side defence can outperform Tesseract on obfuscated text in images.

A two-stage convolutional neural network (CNN)
solution is proposed. Stage 1 trains a 2.46M-parameter CNN on a synthetically
generated corpus of isolated characters rendered with diverse LeetSpeak, homoglyph,
and spacing attacks. Stage 2 reuses the frozen character encoder in a
sliding-window arrangement to classify fixed-length three-letter words drawn
from the full 26³ (17,576-class) vocabulary; the word head is subsequently
fine-tuned after the last convolutional layer of the base classifier is unfrozen.

On a held-out character dataset, the character
(stage 1) model attains 96.7% accuracy versus Tesseract’s 54.2%. For three-character
sequences, the word (stage 2) model achieves 21.7% exact-match accuracy (more
than doubling Tesseract’s 9.6%). These results validate the hypothesis that
image-level de-obfuscation can substantially harden OCR pipelines against
character-level attacks. Although the current scope is limited to uppercase,
fixed-length words, the methodology scales naturally to mixed-case and
variable-length text via sequence decoders and larger and more varied datasets.
The work therefore lays a practical foundation for deployable, modular defences
that _see through_ adversarial text transformations in real-world
moderation and OCR systems.

## Data

The data used in this project is freely available via the following links:

- **Character dataset:** [Drive](https://drive.google.com/drive/folders/1eUaTNW8zVjTArg0JszbCdCEq0tTdx89n?usp=drive_link 'Google Drive')
- **Word dataset:** [Drive](https://drive.google.com/drive/folders/1kygA17GiCeCs8qTeDBEndU6TkXnEu-m7?usp=drive_link 'Google Drive')

## Running Training / Inference Pipelines

See the 3 notebooks in the ` notebooks/` directory.

## Licence

See the LICENSE file
