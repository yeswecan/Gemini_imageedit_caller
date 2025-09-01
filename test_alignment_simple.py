#!/usr/bin/env python3
"""
Simple test of alignment functionality without OpenCV visualization.
"""
from pathlib import Path
import json

def test_basic_functionality():
    """Test if we can at least load and run the basic processing."""
    print("Testing basic image processing functionality...")
    
    # Check if results exist
    results_dir = Path("results")
    if not results_dir.exists():
        print("Error: No results directory found. Please run generate_all_results.py first.")
        return False
    
    # Count existing results
    result_files = list(results_dir.glob("*.png"))
    print(f"Found {len(result_files)} result images")
    
    # Check if we can import our modules
    try:
        from image_processor import ImageProcessor
        print("✓ Successfully imported ImageProcessor")
    except Exception as e:
        print(f"✗ Failed to import ImageProcessor: {e}")
        return False
    
    try:
        # Test basic instantiation
        processor = ImageProcessor()
        print("✓ Successfully created ImageProcessor instance")
        print(f"  Model: {processor.model}")
        print(f"  API URL: {processor.api_url}")
        print(f"  Prompt: {processor.prompt[:50]}...")
        return True
    except Exception as e:
        print(f"✗ Failed to create processor: {e}")
        return False

def test_generation_without_alignment():
    """Test generation without the alignment step."""
    print("\nTesting generation without alignment...")
    
    template_path = Path("characters/Ali_F.png")
    selfie_path = Path("selfies_samples/005_Selfie_12.jpg")
    output_path = Path("test_no_alignment.png")
    
    if not template_path.exists() or not selfie_path.exists():
        print("Error: Test images not found")
        return False
    
    try:
        from image_processor import ImageProcessor
        processor = ImageProcessor()
        
        # Test with alignment disabled
        print(f"Processing {template_path.name} + {selfie_path.name}...")
        result = processor.process_images(
            template_path=template_path,
            selfie_path=selfie_path,
            output_path=output_path,
            align=False  # Disable alignment
        )
        
        print(f"Result: {json.dumps(result, indent=2)}")
        
        if result['success']:
            print(f"✓ Successfully generated image at {output_path}")
            return True
        else:
            print(f"✗ Generation failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running simple alignment tests...\n")
    
    # Test 1: Basic functionality
    if test_basic_functionality():
        print("\n✓ Basic functionality test passed")
    else:
        print("\n✗ Basic functionality test failed")
        exit(1)
    
    # Test 2: Generation without alignment
    # Skip this for now as it would use API credits
    print("\nSkipping generation test to save API credits.")
    print("To test generation, uncomment the test_generation_without_alignment() call.")
    # if test_generation_without_alignment():
    #     print("\n✓ Generation test passed")
    # else:
    #     print("\n✗ Generation test failed")