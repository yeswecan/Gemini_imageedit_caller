#!/usr/bin/env python3
import zipfile
from pathlib import Path
import shutil

zip_path = Path("selfies-id-images-dataset.zip")
samples_dir = Path("selfies_samples")
samples_dir.mkdir(exist_ok=True)

print(f"Extracting 50 sample images from {zip_path.name}...")

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
extracted_count = 0

with zipfile.ZipFile(zip_path, 'r') as zf:
    # Get list of all files in the zip
    all_files = zf.namelist()
    
    # Filter for image files
    image_files = [f for f in all_files if Path(f).suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in the dataset")
    
    # Extract first 50 images
    for i, file_name in enumerate(image_files[:50]):
        # Create a simple numbered filename
        original_name = Path(file_name).name
        output_name = f"{i:03d}_{original_name}"
        output_path = samples_dir / output_name
        
        # Extract the file
        file_data = zf.read(file_name)
        with open(output_path, 'wb') as f:
            f.write(file_data)
        
        print(f"  [{i+1}/50] {output_name}")
        extracted_count += 1

print(f"\nSuccessfully extracted {extracted_count} images to: {samples_dir.absolute()}")
print("\nSample images are ready!")