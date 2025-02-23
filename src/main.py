from image_generation import generate_profanity_images

def main():
    profanities_file = "data/profanities.txt"
    fonts_folder = "fonts"
    output_dir = "data/raw"

    # Step 1: Generate images of profanities using different typefaces
    print("Starting image generation...")
    generate_profanity_images(profanities_file, fonts_folder, output_dir, include_obfuscations=True, multi_cases=True)
    print("Image generation completed.")
    
    # Step 2: Pre-process images and prepare dataset
    print("Preprocessing and preparing images for training...")
    
    # Step 3: Train the CNN model on the processed data
    print("Training CNN model...")
    
    # Step 4: Run OCR pipelines with both Google Vision OCR and Microsoft TrOCR, and text filtering
    print("Running OCR pipelines...")
    
    # Step 5: Evaluate and compare the CNN model against the OCR-based pipelines
    print("Evaluating all pipelines...")

if __name__ == '__main__':
    main()
