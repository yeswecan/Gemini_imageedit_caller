#!/usr/bin/env python3
import subprocess
import sys
import time
from pathlib import Path

print("Starting download of selfies-id-images-dataset (975MB)...")
print("This will download in the background and show progress.")

# Start the download process
cmd = ["kaggle", "datasets", "download", "-d", "tapakah68/selfies-id-images-dataset"]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Monitor the download
start_time = time.time()
while True:
    output = process.stdout.readline()
    if output == '' and process.poll() is not None:
        break
    if output:
        print(output.strip())
        sys.stdout.flush()

elapsed = time.time() - start_time
print(f"\nDownload completed in {elapsed:.1f} seconds")

# Check if zip file exists
zip_files = list(Path(".").glob("*.zip"))
if zip_files:
    print(f"Downloaded: {zip_files[0].name}")
    print(f"File size: {zip_files[0].stat().st_size / 1024 / 1024:.1f} MB")