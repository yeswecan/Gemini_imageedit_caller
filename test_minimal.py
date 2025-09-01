#!/usr/bin/env python3
"""
Minimal test without face alignment dependencies.
Creates visualizations using PIL instead of OpenCV.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json

def test_server_without_alignment():
    """Test the server endpoint without alignment."""
    print("Testing server (without face alignment)...")
    
    # First, let's modify the server to work without InsightFace
    server_py = Path("server.py")
    server_content = server_py.read_text()
    
    # Create a version without alignment
    modified_server = server_content.replace(
        "from image_processor import ImageProcessor",
        "from image_processor_no_align import ImageProcessor"
    )
    
    # Save modified processor without alignment
    print("Creating processor without alignment dependencies...")
    create_processor_without_alignment()
    
    print("\nYou can now test the server by:")
    print("1. Run: source venv/bin/activate && python server_no_align.py")
    print("2. In another terminal: python test_server.py")

def create_processor_without_alignment():
    """Create a version of image_processor without face_alignment dependency."""
    processor_content = '''#!/usr/bin/env python3
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
'''
    
    with open("image_processor_no_align.py", "w") as f:
        f.write(processor_content)
    print("Created image_processor_no_align.py")
    
    # Create server without alignment
    server_content = Path("server.py").read_text()
    server_content = server_content.replace(
        "from image_processor import ImageProcessor",
        "from image_processor_no_align import ImageProcessor"
    )
    
    with open("server_no_align.py", "w") as f:
        f.write(server_content)
    print("Created server_no_align.py")

def draw_face_landmarks_pil():
    """Draw landmarks on existing images using PIL to verify they exist."""
    print("\nVisualizing existing results with PIL...")
    
    # Pick a result that exists
    result_path = Path("results/Ali_F_005_Selfie_12_result.png")
    if not result_path.exists():
        print(f"Result not found: {result_path}")
        return
    
    # Open and create visualization
    img = Image.open(result_path)
    draw = ImageDraw.Draw(img)
    
    # Draw some mock landmarks (since we can't detect without InsightFace)
    # This is just to show the visualization would work
    width, height = img.size
    
    # Mock eye positions
    left_eye = (width * 0.35, height * 0.4)
    right_eye = (width * 0.65, height * 0.4)
    mouth_center = (width * 0.5, height * 0.7)
    
    # Acid green color
    acid_green = (127, 255, 0)
    
    # Draw circles
    for point, label in [(left_eye, "L_Eye"), (right_eye, "R_Eye"), (mouth_center, "Mouth")]:
        x, y = int(point[0]), int(point[1])
        # Draw circle
        draw.ellipse([x-5, y-5, x+5, y+5], fill=acid_green, outline=acid_green)
        draw.ellipse([x-7, y-7, x+7, y+7], outline=acid_green, width=2)
        # Draw label
        draw.text((x+10, y-10), label, fill=acid_green)
    
    # Draw line between eyes
    draw.line([left_eye, right_eye], fill=acid_green, width=2)
    
    # Draw lines from eyes to mouth
    eye_center = ((left_eye[0] + right_eye[0])/2, (left_eye[1] + right_eye[1])/2)
    draw.line([eye_center, mouth_center], fill=acid_green, width=2)
    
    # Save
    output_path = Path("test_landmarks_visualization.png")
    img.save(output_path)
    print(f"Saved mock visualization to: {output_path}")
    print("Note: These are mock landmarks for visualization only!")
    
    return True

if __name__ == "__main__":
    print("Running minimal tests without face alignment...\n")
    
    # Create no-alignment versions
    test_server_without_alignment()
    
    # Create visualization with PIL
    draw_face_landmarks_pil()
    
    print("\nTo properly test face alignment, you would need to:")
    print("1. Install opencv-python and insightface in a compatible Python environment")
    print("2. Run the full test_alignment_visual.py script")
    print("\nFor now, you can test the API without alignment using:")
    print("python server_no_align.py")