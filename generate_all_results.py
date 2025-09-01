#!/usr/bin/env python3
import os
import base64
import requests
import json
import time
from pathlib import Path
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
if not API_KEY:
    env_path = Path(".env")
    if env_path.exists():
        API_KEY = env_path.read_text().strip()

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-image-preview"

# Read prompt
prompt_path = Path("prompt.md")
PROMPT = prompt_path.read_text().strip()

# Thread-safe counter for rate limiting
request_lock = threading.Lock()
last_request_time = 0

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def make_api_request(character_image_path, selfie_image_path, max_retries=3):
    """Make API request with retry logic and rate limiting"""
    global last_request_time
    
    character_b64 = encode_image_to_base64(character_image_path)
    selfie_b64 = encode_image_to_base64(selfie_image_path)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Image Edit Tester"
    }
    
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
            # Rate limiting
            with request_lock:
                current_time = time.time()
                time_since_last = current_time - last_request_time
                if time_since_last < 0.5:  # Max 2 requests per second
                    time.sleep(0.5 - time_since_last)
                last_request_time = time.time()
            
            start_time = time.time()
            response = requests.post(API_URL, headers=headers, json=data, timeout=60)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return result, elapsed_time
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                if attempt == max_retries - 1:
                    return {"error": error_msg}, elapsed_time
                time.sleep(2 ** attempt)
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"Exception: {str(e)}"
            if attempt == max_retries - 1:
                return {"error": error_msg}, elapsed_time
            time.sleep(2 ** attempt)

def extract_and_save_image(response, output_path):
    """Extract image from response and save it"""
    try:
        if "error" in response:
            return False, response["error"]
            
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
                        return True, None
                    else:
                        return False, f"Image URL is not base64 encoded. Response: {json.dumps(response, indent=2)}"
                else:
                    return False, f"Invalid image data structure. Response: {json.dumps(response, indent=2)}"
            else:
                # Return full response when no images found
                return False, f"No images found. Full response: {json.dumps(response, indent=2)}"
        else:
            return False, f"Invalid response structure. Response: {json.dumps(response, indent=2)}"
            
    except Exception as e:
        return False, f"Error extracting image: {str(e)}. Response: {json.dumps(response, indent=2) if 'response' in locals() else 'No response'}"

def process_image_pair(character_path, selfie_path, results_dir):
    """Process a single character-selfie pair"""
    output_name = f"{character_path.stem}_{selfie_path.stem}_result.png"
    output_path = results_dir / output_name
    
    print(f"Processing: {character_path.name} + {selfie_path.name}")
    
    response, elapsed_time = make_api_request(character_path, selfie_path)
    success, error_msg = extract_and_save_image(response, output_path)
    
    result = {
        "character": character_path.name,
        "selfie": selfie_path.name,
        "output": output_name if success else None,
        "time": elapsed_time,
        "error": error_msg,
        "success": success
    }
    
    if success:
        print(f"✓ Completed: {output_name} ({elapsed_time:.2f}s)")
    else:
        print(f"✗ Failed: {character_path.name} + {selfie_path.name} - {error_msg}")
    
    return result

def generate_markdown_table(results, characters, selfies):
    """Generate markdown table from results"""
    md_content = "# Image Generation Results\n\n"
    md_content += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"Model: {MODEL}\n\n"
    md_content += f"Prompt: {PROMPT}\n\n"
    
    # Create results lookup
    results_dict = {}
    for r in results:
        key = (r["character"], r["selfie"])
        results_dict[key] = r
    
    # Generate table header with selfie images
    md_content += "| Character |"
    for selfie in selfies:
        # Show the selfie image in the header
        selfie_img = f"![{selfie.name}](selfies_samples/{selfie.name})"
        md_content += f" {selfie_img}<br>**{selfie.stem}** |"
    md_content += "\n"
    
    md_content += "|-----------|"
    for _ in selfies:
        md_content += ":---:|"  # Center align columns
    md_content += "\n"
    
    # Generate table rows with character images
    for character in characters:
        # Show the character image in the first column
        char_img = f"![{character.name}](characters/{character.name})"
        md_content += f"| {char_img}<br>**{character.stem}** |"
        
        for selfie in selfies:
            key = (character.name, selfie.name)
            result = results_dict.get(key, {})
            
            if result.get("success"):
                # Success - show image and time
                # Use relative path for GitHub compatibility
                img_path = f"results/{result['output']}"
                cell_content = f"![{result['output']}]({img_path})<br>{result['time']:.2f}s"
            else:
                # Failure - show error
                error = result.get("error", "Unknown error")
                # Don't truncate errors anymore to see full response
                # Escape HTML characters for markdown
                error = error.replace("<", "&lt;").replace(">", "&gt;")
                # Make JSON more readable by adding line breaks
                if "Full response:" in error or "Response:" in error:
                    error = error.replace('", "', '",<br>"').replace('{', '{<br>').replace('}', '<br>}')
                cell_content = f"❌ Error<br><details><summary>Click to expand</summary><pre>{error}</pre></details><br>{result.get('time', 0):.2f}s"
            
            md_content += f" {cell_content} |"
        
        md_content += "\n"
    
    # Add statistics
    total = len(results)
    successful = sum(1 for r in results if r.get("success"))
    failed = total - successful
    avg_time = sum(r.get("time", 0) for r in results) / total if total > 0 else 0
    
    md_content += f"\n## Statistics\n\n"
    md_content += f"- Total requests: {total}\n"
    md_content += f"- Successful: {successful} ({successful/total*100:.1f}%)\n"
    md_content += f"- Failed: {failed} ({failed/total*100:.1f}%)\n"
    md_content += f"- Average time: {avg_time:.2f}s\n"
    
    return md_content

def main():
    """Process all images and generate markdown table"""
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Get all characters and selfies
    characters_dir = Path("characters")
    selfies_dir = Path("selfies_samples")
    
    characters = sorted(list(characters_dir.glob("*.png")))
    selfies = sorted(list(selfies_dir.glob("*.jpg")))
    
    print(f"Found {len(characters)} characters and {len(selfies)} selfies")
    print(f"Total combinations to process: {len(characters) * len(selfies)}")
    print()
    
    # Process all combinations
    results = []
    tasks = []
    
    # Use ThreadPoolExecutor for parallel processing
    max_workers = 3  # Limit concurrent requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for character in characters:
            for selfie in selfies:
                future = executor.submit(process_image_pair, character, selfie, results_dir)
                tasks.append(future)
        
        # Collect results as they complete
        for future in as_completed(tasks):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Task failed with exception: {e}")
    
    # Generate markdown table
    print("\nGenerating markdown table...")
    md_content = generate_markdown_table(results, characters, selfies)
    
    # Save markdown file
    md_path = Path("results_table.md")
    md_path.write_text(md_content)
    print(f"Markdown table saved to: {md_path}")
    
    # Print summary
    successful = sum(1 for r in results if r.get("success"))
    print(f"\nCompleted! {successful}/{len(results)} successful generations")

if __name__ == "__main__":
    main()