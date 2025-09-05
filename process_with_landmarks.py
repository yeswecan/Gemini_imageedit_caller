#!/usr/bin/env python3
"""
Process images with landmark visualization for debugging.
"""
import cv2
import numpy as np
from pathlib import Path
from face_alignment_unified import UnifiedFaceAligner as FaceAligner
from image_processor import ImageProcessor
import time
import json

def add_landmarks_to_image(image_path, output_path, aligner=None):
    """
    Add landmark visualization to an image.
    If no face detected, add "NO FACE DETECTED" text.
    """
    if aligner is None:
        aligner = FaceAligner()
    
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read image: {image_path}")
        return False
    
    # Detect landmarks
    result = aligner.detect_landmarks(image_path)
    
    # Acid green color
    acid_green = (0, 255, 127)
    
    if result['success']:
        # Draw all 5 landmarks
        for i, point in enumerate(result['landmarks']):
            x, y = int(point[0]), int(point[1])
            cv2.circle(img, (x, y), 5, acid_green, -1)
            cv2.circle(img, (x, y), 7, acid_green, 2)
        
        # Draw eye center with larger circle
        eye_center = result['eye_center']
        x, y = int(eye_center[0]), int(eye_center[1])
        cv2.circle(img, (x, y), 10, acid_green, -1)
        cv2.circle(img, (x, y), 12, acid_green, 2)
        
        # Draw mouth center with larger circle
        mouth_center = result['mouth_center']
        x, y = int(mouth_center[0]), int(mouth_center[1])
        cv2.circle(img, (x, y), 10, acid_green, -1)
        cv2.circle(img, (x, y), 12, acid_green, 2)
        
        # Draw line between eye and mouth center
        cv2.line(img,
                 (int(eye_center[0]), int(eye_center[1])),
                 (int(mouth_center[0]), int(mouth_center[1])),
                 acid_green, 3)
    else:
        # Add "NO FACE DETECTED" text
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "NO FACE DETECTED"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, 1.5, 3)
        
        # Center text
        text_x = (w - text_width) // 2
        text_y = h // 2
        
        # Draw text with outline for visibility
        cv2.putText(img, text, (text_x, text_y), font, 1.5, (0, 0, 0), 5)  # Black outline
        cv2.putText(img, text, (text_x, text_y), font, 1.5, acid_green, 3)  # Green text
    
    # Save image
    cv2.imwrite(str(output_path), img)
    return result['success']

def process_single_pair_with_landmarks(template_path, selfie_path, output_dir):
    """
    Process a single template-selfie pair with full landmark visualization and timing.
    """
    print(f"\nProcessing pair:")
    print(f"  Template: {template_path}")
    print(f"  Selfie: {selfie_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize processors
    aligner = FaceAligner()
    processor = ImageProcessor()
    
    # Step 1: Add landmarks to template
    template_with_landmarks = output_dir / f"template_landmarks_{template_path.name}"
    print("\n1. Adding landmarks to template...")
    template_has_face = add_landmarks_to_image(template_path, template_with_landmarks, aligner)
    print(f"   Template face detection: {'SUCCESS' if template_has_face else 'FAILED'}")
    
    # Step 2: Add landmarks to selfie  
    selfie_with_landmarks = output_dir / f"selfie_landmarks_{selfie_path.name}"
    print("\n2. Adding landmarks to selfie...")
    selfie_has_face = add_landmarks_to_image(selfie_path, selfie_with_landmarks, aligner)
    print(f"   Selfie face detection: {'SUCCESS' if selfie_has_face else 'FAILED'}")
    
    # Step 3: Generate face swap
    print("\n3. Generating face swap...")
    start_time = time.time()
    
    # Use processor to generate without alignment first
    temp_output = output_dir / "temp_generated.png"
    result = processor.process_images(
        template_path=template_path,
        selfie_path=selfie_path,
        output_path=temp_output,
        align=False  # No alignment yet
    )
    
    generation_time = time.time() - start_time
    
    if not result['success']:
        print(f"   Generation failed: {result.get('error', 'Unknown error')}")
        return None
    
    print(f"   Generation time: {generation_time:.2f}s")
    print(f"   API retries: {result['retries']}")
    
    # Step 4: Add landmarks to generated image
    generated_with_landmarks = output_dir / "generated_landmarks.png"
    print("\n4. Adding landmarks to generated image...")
    generated_has_face = add_landmarks_to_image(temp_output, generated_with_landmarks, aligner)
    print(f"   Generated face detection: {'SUCCESS' if generated_has_face else 'FAILED'}")
    
    # Step 5: Align if both template and generated have faces
    if template_has_face and generated_has_face:
        print("\n5. Aligning generated image to template...")
        
        # Get template dimensions
        template_img = cv2.imread(str(template_path))
        h_t, w_t = template_img.shape[:2]
        print(f"   Template dimensions: {w_t}x{h_t}")
        
        # Align
        aligned_output = output_dir / "aligned_result.png"
        align_result = aligner.align_image(temp_output, template_path, aligned_output)
        
        if align_result['success']:
            print(f"   Alignment successful!")
            print(f"   Scale: {align_result['scale']:.3f}")
            print(f"   Angle: {align_result['angle']:.1f} degrees")
            
            # Verify dimensions
            aligned_img = cv2.imread(str(aligned_output))
            h_a, w_a = aligned_img.shape[:2]
            print(f"   Aligned dimensions: {w_a}x{h_a}")
            
            if w_a == w_t and h_a == h_t:
                print("   ✓ Dimensions match template!")
            else:
                print("   ✗ Dimensions don't match!")
            
            # Add landmarks to aligned result
            aligned_with_landmarks = output_dir / "aligned_with_landmarks.png"
            print("\n6. Adding landmarks to aligned result...")
            add_landmarks_to_image(aligned_output, aligned_with_landmarks, aligner)
        else:
            print(f"   Alignment failed: {align_result['error']}")
    else:
        print("\n5. Skipping alignment (missing face detection)")
    
    # Create comparison image
    print("\n7. Creating comparison image...")
    create_comparison_image(
        template_with_landmarks,
        selfie_with_landmarks,
        generated_with_landmarks,
        aligned_with_landmarks if template_has_face and generated_has_face else None,
        output_dir / "comparison.png"
    )
    
    # Return processing info
    return {
        'generation_time': generation_time,
        'retries': result['retries'],
        'template_has_face': template_has_face,
        'generated_has_face': generated_has_face,
        'aligned': template_has_face and generated_has_face
    }

def create_comparison_image(template_path, selfie_path, generated_path, aligned_path, output_path):
    """Create a side-by-side comparison of all images."""
    # Read images
    template = cv2.imread(str(template_path))
    selfie = cv2.imread(str(selfie_path))
    generated = cv2.imread(str(generated_path))
    
    # Resize to consistent height
    height = 500
    images = []
    labels = ["Template", "Selfie", "Generated"]
    
    for img in [template, selfie, generated]:
        h, w = img.shape[:2]
        new_w = int(w * height / h)
        resized = cv2.resize(img, (new_w, height))
        images.append(resized)
    
    if aligned_path and Path(aligned_path).exists():
        aligned = cv2.imread(str(aligned_path))
        h, w = aligned.shape[:2]
        new_w = int(w * height / h)
        resized = cv2.resize(aligned, (new_w, height))
        images.append(resized)
        labels.append("Aligned")
    
    # Create canvas
    total_width = sum(img.shape[1] for img in images) + 10 * (len(images) + 1)
    canvas = np.ones((height + 50, total_width, 3), dtype=np.uint8) * 255
    
    # Place images
    x = 10
    for i, (img, label) in enumerate(zip(images, labels)):
        canvas[40:40+height, x:x+img.shape[1]] = img
        
        # Add label
        cv2.putText(canvas, label, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        x += img.shape[1] + 10
    
    # Save
    cv2.imwrite(str(output_path), canvas)
    print(f"   Saved comparison to: {output_path}")

if __name__ == "__main__":
    # Test with a single pair
    template = Path("characters/Ali_F.png")
    selfie = Path("selfies_samples/005_Selfie_12.jpg")
    
    info = process_single_pair_with_landmarks(template, selfie, "test_landmarks_output")
    
    if info:
        print("\n" + "="*50)
        print("SUMMARY:")
        print(f"Generation time: {info['generation_time']:.2f}s")
        print(f"API retries: {info['retries']}")
        print(f"Template has face: {info['template_has_face']}")
        print(f"Generated has face: {info['generated_has_face']}")
        print(f"Alignment performed: {info['aligned']}")
        print("="*50)