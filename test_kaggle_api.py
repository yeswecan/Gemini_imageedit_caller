import os
import shutil
from pathlib import Path

# Set up Kaggle credentials
kaggle_dir = Path.home() / '.kaggle'
kaggle_dir.mkdir(exist_ok=True)

# Copy kaggle.json to the correct location
kaggle_json_home = Path.home() / 'kaggle.json'
kaggle_json_dest = kaggle_dir / 'kaggle.json'

if kaggle_json_home.exists() and not kaggle_json_dest.exists():
    shutil.copy(kaggle_json_home, kaggle_json_dest)
    os.chmod(kaggle_json_dest, 0o600)

# Import and test kaggle API
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    api = KaggleApi()
    api.authenticate()
    print("✓ Kaggle API authenticated successfully!")
    
    # Try to get dataset info
    dataset = "tapakah68/selfies-id-images-dataset"
    print(f"\nGetting info for dataset: {dataset}")
    
    # List dataset files
    files = api.dataset_list_files(dataset)
    print(f"\nDataset files:")
    for i, file in enumerate(files.files[:10]):  # Show first 10 files
        print(f"  {i+1}. {file.name}")
    
    if len(files.files) > 10:
        print(f"  ... and {len(files.files) - 10} more files")
    
    print(f"\nTotal files: {len(files.files)}")
    
    # Try to download just one file as a test
    print("\nTrying to download the first file as a test...")
    if files.files:
        first_file = files.files[0]
        api.dataset_download_file(dataset, first_file.name, path="./test_download")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nPossible issues:")
    print("1. Check if you've accepted the dataset's terms on Kaggle website")
    print("2. Visit: https://www.kaggle.com/datasets/tapakah68/selfies-id-images-dataset")
    print("3. Click 'Download' button to accept any terms if required")