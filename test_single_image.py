#!/usr/bin/env python3
import os
import base64
import requests
import json
import time
from pathlib import Path
from PIL import Image
import io

# Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
if not API_KEY:
    # Read from .env file if not in environment
    env_path = Path(".env")
    if env_path.exists():
        API_KEY = env_path.read_text().strip()

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-image-preview"

# Read prompt
prompt_path = Path("prompt.md")
PROMPT = prompt_path.read_text().strip()

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def make_api_request(character_image_path, selfie_image_path, max_retries=3):
    """Make API request with retry logic"""
    character_b64 = encode_image_to_base64(character_image_path)
    selfie_b64 = encode_image_to_base64(selfie_image_path)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Image Edit Tester"
    }
    
    # Construct the request
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{character_b64}",
                            "detail": "auto"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{selfie_b64}",
                            "detail": "auto"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}...")
            start_time = time.time()
            
            response = requests.post(API_URL, headers=headers, json=data, timeout=60)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"Success! Response received in {elapsed_time:.2f} seconds")
                return result, elapsed_time
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                print(error_msg)
                if attempt == max_retries - 1:
                    return {"error": error_msg}, elapsed_time
                time.sleep(2 ** attempt)  # Exponential backoff
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"Exception: {str(e)}"
            print(error_msg)
            if attempt == max_retries - 1:
                return {"error": error_msg}, elapsed_time
            time.sleep(2 ** attempt)

def extract_and_save_image(response, output_path):
    """Extract image from response and save it"""
    try:
        # Check if response contains an error
        if "error" in response:
            print(f"Error in response: {response['error']}")
            return False
            
        # Extract content from response
        if "choices" in response and len(response["choices"]) > 0:
            message = response["choices"][0]["message"]
            
            # Check if message has images array
            if "images" in message and len(message["images"]) > 0:
                # Get the first image
                image_data = message["images"][0]
                if "image_url" in image_data and "url" in image_data["image_url"]:
                    image_url = image_data["image_url"]["url"]
                    
                    # Extract base64 data
                    if "base64," in image_url:
                        base64_data = image_url.split("base64,")[1]
                        
                        # Decode and save
                        image_bytes = base64.b64decode(base64_data)
                        image = Image.open(io.BytesIO(image_bytes))
                        image.save(output_path)
                        print(f"Image saved to: {output_path}")
                        return True
                    else:
                        print("Image URL is not base64 encoded")
                        return False
                else:
                    print("Invalid image data structure")
                    return False
            else:
                print("No images found in response")
                print(f"Message: {json.dumps(message, indent=2)[:500]}...")
                return False
        else:
            print("Invalid response structure")
            print(f"Response: {json.dumps(response, indent=2)[:500]}...")
            return False
            
    except Exception as e:
        print(f"Error extracting image: {str(e)}")
        return False

def main():
    """Test with single images"""
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Get first character and selfie
    characters_dir = Path("characters")
    selfies_dir = Path("selfies_samples")
    
    character_files = sorted(list(characters_dir.glob("*.png")))
    selfie_files = sorted(list(selfies_dir.glob("*.jpg")))
    
    if not character_files:
        print("No character files found!")
        return
    
    if not selfie_files:
        print("No selfie files found!")
        return
    
    # Test with first of each
    character_path = character_files[0]
    selfie_path = selfie_files[0]
    
    print(f"Testing with:")
    print(f"  Character: {character_path.name}")
    print(f"  Selfie: {selfie_path.name}")
    print(f"  Prompt: {PROMPT}")
    print()
    
    # Make API request
    response, elapsed_time = make_api_request(character_path, selfie_path)
    
    # Save result
    output_name = f"{character_path.stem}_{selfie_path.stem}_result.png"
    output_path = results_dir / output_name
    
    if extract_and_save_image(response, output_path):
        print(f"\nTest completed successfully!")
        print(f"Time taken: {elapsed_time:.2f} seconds")
    else:
        print(f"\nTest failed!")
        print(f"Response: {json.dumps(response, indent=2)}")

if __name__ == "__main__":
    main()