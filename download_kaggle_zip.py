#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import zipfile

# Set up Kaggle credentials
kaggle_dir = Path.home() / '.kaggle'
kaggle_dir.mkdir(exist_ok=True)

kaggle_json_home = Path.home() / 'kaggle.json'
kaggle_json_dest = kaggle_dir / 'kaggle.json'

if kaggle_json_home.exists() and not kaggle_json_dest.exists():
    shutil.copy(kaggle_json_home, kaggle_json_dest)
    os.chmod(kaggle_json_dest, 0o600)

print("Downloading dataset using Kaggle CLI...")
print("This might take a few minutes depending on your internet connection.")

# Use Kaggle CLI directly
dataset = "tapakah68/selfies-id-images-dataset"
download_dir = "./kaggle_download"

# Create download directory
Path(download_dir).mkdir(exist_ok=True)

# Download using subprocess (more reliable for large downloads)
import subprocess

cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", download_dir]
print(f"Running: {' '.join(cmd)}")

try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error downloading: {result.stderr}")
        print("\nPlease make sure:")
        print("1. You have accepted the dataset terms on Kaggle")
        print("2. Visit: https://www.kaggle.com/datasets/tapakah68/selfies-id-images-dataset")
        exit(1)
    
    print("Download complete!")
    
    # Find the zip file
    zip_files = list(Path(download_dir).glob("*.zip"))
    if not zip_files:
        print("No zip file found!")
        exit(1)
    
    zip_path = zip_files[0]
    print(f"\nExtracting from: {zip_path.name}")
    
    # Create samples directory
    samples_dir = Path("./selfies_samples")
    samples_dir.mkdir(exist_ok=True)
    
    # Extract only the first 50 images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    extracted_count = 0
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get list of image files in the zip
        image_files = [f for f in zf.namelist() if Path(f).suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images in the dataset")
        
        # Extract up to 50 images
        for i, file_name in enumerate(image_files[:50]):
            # Extract to samples directory with a simple numbered name
            file_data = zf.read(file_name)
            output_name = f"{i:03d}_{Path(file_name).name}"
            output_path = samples_dir / output_name
            
            with open(output_path, 'wb') as f:
                f.write(file_data)
            
            print(f"  [{i+1}/50] Extracted: {output_name}")
            extracted_count += 1
    
    print(f"\nSuccessfully extracted {extracted_count} images to: {samples_dir.absolute()}")
    
    # Clean up
    print("\nCleaning up...")
    shutil.rmtree(download_dir)
    print("Done!")
    
except FileNotFoundError:
    print("Error: 'kaggle' command not found!")
    print("Please make sure kaggle is installed in the virtual environment")
    print("Run: pip install kaggle")