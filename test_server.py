#!/usr/bin/env python3
"""
Simple test script for the face swap API server.
"""
import requests
import sys
from pathlib import Path

def test_health_check(base_url="http://localhost:5000"):
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✓ Health check passed:", response.json())
            return True
        else:
            print("✗ Health check failed:", response.status_code)
            return False
    except Exception as e:
        print("✗ Cannot connect to server:", str(e))
        return False

def test_face_swap(template_path, selfie_path, output_path, base_url="http://localhost:5000"):
    """Test the face swap endpoint"""
    try:
        with open(template_path, 'rb') as template_file, open(selfie_path, 'rb') as selfie_file:
            files = {
                'template': ('template.png', template_file, 'image/png'),
                'selfie': ('selfie.jpg', selfie_file, 'image/jpeg')
            }
            
            print(f"Sending request to {base_url}/swap_face...")
            response = requests.post(f"{base_url}/swap_face", files=files)
            
            if response.status_code == 200:
                # Save the result
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"✓ Face swap successful! Result saved to: {output_path}")
                print(f"  Generation time: {response.headers.get('X-Generation-Time', 'N/A')}s")
                print(f"  Alignment status: {response.headers.get('X-Alignment-Status', 'N/A')}")
                print(f"  Total time: {response.headers.get('X-Total-Time', 'N/A')}s")
                print(f"  Retries: {response.headers.get('X-Retries', 'N/A')}")
                return True
            else:
                print(f"✗ Face swap failed with status {response.status_code}")
                print(f"  Error: {response.text}")
                return False
    except Exception as e:
        print(f"✗ Request failed: {str(e)}")
        return False

def main():
    """Run server tests"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python test_server.py [template_path] [selfie_path] [output_path]")
        print("If no arguments provided, uses default test images")
        return
    
    # Check if server is running
    if not test_health_check():
        print("\nPlease start the server first: python server.py")
        return
    
    # Use provided paths or defaults
    if len(sys.argv) == 4:
        template_path = sys.argv[1]
        selfie_path = sys.argv[2]
        output_path = sys.argv[3]
    else:
        # Use default test images
        template_path = "characters/Ali_F.png"
        selfie_path = "selfies_samples/005_Selfie_12.jpg"
        output_path = "test_result.png"
        print(f"\nUsing default test images:")
        print(f"  Template: {template_path}")
        print(f"  Selfie: {selfie_path}")
    
    # Check if files exist
    if not Path(template_path).exists():
        print(f"✗ Template file not found: {template_path}")
        return
    if not Path(selfie_path).exists():
        print(f"✗ Selfie file not found: {selfie_path}")
        return
    
    # Test face swap
    print("\nTesting face swap endpoint...")
    test_face_swap(template_path, selfie_path, output_path)

if __name__ == "__main__":
    main()