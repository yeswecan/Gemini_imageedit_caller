import os
import json
import zipfile
from pathlib import Path
import shutil

# Set up Kaggle credentials
kaggle_dir = Path.home() / '.kaggle'
kaggle_dir.mkdir(exist_ok=True)

# Copy kaggle.json to the correct location if not already there
kaggle_json_home = Path.home() / 'kaggle.json'
kaggle_json_dest = kaggle_dir / 'kaggle.json'

if kaggle_json_home.exists() and not kaggle_json_dest.exists():
    shutil.copy(kaggle_json_home, kaggle_json_dest)
    os.chmod(kaggle_json_dest, 0o600)

# Import kaggle after setting up credentials
import kaggle

# Download the dataset
dataset_name = "tapakah68/selfies-id-images-dataset"
download_path = Path("./kaggle_download")
download_path.mkdir(exist_ok=True)

print(f"Downloading dataset: {dataset_name}")
kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)

# Find image files in the downloaded dataset
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
image_files = []

for file_path in download_path.rglob('*'):
    if file_path.suffix.lower() in image_extensions:
        image_files.append(file_path)

print(f"Found {len(image_files)} image files")

# Copy first 50 images to a samples directory
samples_dir = Path("./selfies_samples")
samples_dir.mkdir(exist_ok=True)

for i, image_path in enumerate(image_files[:50]):
    if i >= 50:
        break
    
    # Copy the image to samples directory
    dest_path = samples_dir / f"{i:03d}_{image_path.name}"
    shutil.copy(image_path, dest_path)
    print(f"Copied: {image_path.name} -> {dest_path.name}")

print(f"\nSuccessfully copied {min(50, len(image_files))} images to {samples_dir}")

# Clean up the full download
shutil.rmtree(download_path)
print("Cleaned up temporary download directory")