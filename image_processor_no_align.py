#!/usr/bin/env python3
"""
Image processor without face alignment - for testing purposes.
"""
import os
import base64
import requests
import json
import time
import logging
from pathlib import Path
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, api_key=None, model="google/gemini-2.5-flash-image-preview"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            env_path = Path(".env")
            if env_path.exists():
                self.api_key = env_path.read_text().strip()
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model
        
        prompt_path = Path("prompt.md")
        self.prompt = prompt_path.read_text().strip()
    
    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def make_api_request(self, template_path, selfie_path, max_retries=3):
        template_b64 = self.encode_image_to_base64(template_path)
        selfie_b64 = self.encode_image_to_base64(selfie_path)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Face Swap Processor"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{template_b64}",
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
                start_time = time.time()
                response = requests.post(self.api_url, headers=headers, json=data, timeout=60)
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    return response.json(), elapsed_time, attempt
                else:
                    error_msg = f"Error {response.status_code}: {response.text}"
                    if attempt == max_retries - 1:
                        return {"error": error_msg}, elapsed_time, attempt
                    time.sleep(2 ** attempt)
                    
            except Exception as e:
                elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
                error_msg = f"Exception: {str(e)}"
                if attempt == max_retries - 1:
                    return {"error": error_msg}, elapsed_time, attempt
                time.sleep(2 ** attempt)
    
    def extract_image_from_response(self, response):
        try:
            if "error" in response:
                return None, response["error"]
                
            if "choices" in response and len(response["choices"]) > 0:
                message = response["choices"][0]["message"]
                
                if "images" in message and len(message["images"]) > 0:
                    image_data = message["images"][0]
                    if "image_url" in image_data and "url" in image_data["image_url"]:
                        image_url = image_data["image_url"]["url"]
                        
                        if "base64," in image_url:
                            base64_data = image_url.split("base64,")[1]
                            image_bytes = base64.b64decode(base64_data)
                            image = Image.open(io.BytesIO(image_bytes))
                            return image, None
                        else:
                            return None, "Image URL is not base64 encoded"
                    else:
                        return None, "Invalid image data structure"
                else:
                    return None, "No images found in response"
            else:
                return None, "Invalid response structure"
                
        except Exception as e:
            return None, f"Error extracting image: {str(e)}"
    
    def process_images(self, template_path, selfie_path, output_path, align=True):
        start_time = time.time()
        result = {
            'success': False,
            'generation_time': 0,
            'alignment_time': 0,
            'total_time': 0,
            'retries': 0
        }
        
        # Generate face swap
        response, gen_time, retries = self.make_api_request(template_path, selfie_path)
        result['generation_time'] = gen_time
        result['retries'] = retries
        
        # Extract generated image
        generated_image, error = self.extract_image_from_response(response)
        
        if generated_image is None:
            result['error'] = f'Generation failed: {error}'
            result['total_time'] = time.time() - start_time
            return result
        
        # Save without alignment
        generated_image.save(str(output_path))
        result['success'] = True
        result['output_path'] = str(output_path)
        
        if align:
            result['alignment_error'] = 'Alignment disabled in test mode'
        
        result['total_time'] = time.time() - start_time
        return result

_processor = None

def get_processor():
    global _processor
    if _processor is None:
        _processor = ImageProcessor()
    return _processor
