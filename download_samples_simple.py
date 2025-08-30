import os
import shutil
from pathlib import Path

# Set up Kaggle credentials
kaggle_dir = Path.home() / '.kaggle'
kaggle_dir.mkdir(exist_ok=True)

# Copy kaggle.json to the correct location if not already there
kaggle_json_home = Path.home() / 'kaggle.json'
kaggle_json_dest = kaggle_dir / 'kaggle.json'

if kaggle_json_home.exists() and not kaggle_json_dest.exists():
    shutil.copy(kaggle_json_home, kaggle_json_dest)
    os.chmod(kaggle_json_dest, 0o600)
    print("Copied kaggle.json to ~/.kaggle/")

# Import kaggle after setting up credentials
try:
    import kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("Please install kaggle: pip install kaggle")
    exit(1)

# Initialize API
api = KaggleApi()
api.authenticate()

# Download the dataset
dataset_name = "tapakah68/selfies-id-images-dataset"
download_path = Path("./kaggle_download")
download_path.mkdir(exist_ok=True)

print(f"Downloading dataset: {dataset_name}")
print("This may take a few minutes...")

try:
    # Download with progress bar
    api.dataset_download_files(dataset_name, path=download_path, unzip=True, quiet=False)
    print("Download complete!")
    
    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = []
    
    print("\nSearching for image files...")
    for file_path in download_path.rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    print(f"Found {len(image_files)} image files")
    
    if image_files:
        # Create samples directory
        samples_dir = Path("./selfies_samples")
        samples_dir.mkdir(exist_ok=True)
        
        # Copy first 50 images
        num_to_copy = min(50, len(image_files))
        print(f"\nCopying {num_to_copy} sample images...")
        
        for i, image_path in enumerate(image_files[:num_to_copy]):
            dest_path = samples_dir / f"{i:03d}_{image_path.name}"
            shutil.copy(image_path, dest_path)
            print(f"  [{i+1}/{num_to_copy}] {image_path.name}")
        
        print(f"\nSuccess! {num_to_copy} images saved to: {samples_dir.absolute()}")
        
        # Clean up
        print("\nCleaning up temporary files...")
        shutil.rmtree(download_path)
        print("Done!")
    else:
        print("No image files found in the dataset!")
        
except Exception as e:
    print(f"Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure your kaggle.json has valid credentials")
    print("2. Check if you accepted the dataset's terms on Kaggle website")
    print("3. Try downloading manually: kaggle datasets download -d tapakah68/selfies-id-images-dataset")