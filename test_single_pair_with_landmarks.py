#!/usr/bin/env python3
"""
Generate a single face swap pair and visualize landmarks on both input and output.
"""
import cv2
import numpy as np
from pathlib import Path
from image_processor import ImageProcessor
from face_alignment import FaceAligner

def test_single_pair():
    """Generate one face swap and show landmarks on both images."""
    print("Testing single face swap with landmark visualization...")
    
    # Use a specific pair
    template_path = Path("characters/Ali_F.png")
    selfie_path = Path("selfies_samples/005_Selfie_12.jpg")
    output_path = Path("test_single_swap_result.png")
    
    print(f"Template: {template_path}")
    print(f"Selfie: {selfie_path}")
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Process images WITH alignment
    print("\nGenerating face swap with alignment...")
    result = processor.process_images(
        template_path=template_path,
        selfie_path=selfie_path,
        output_path=output_path,
        align=True
    )
    
    if result['success']:
        print(f"✓ Face swap successful!")
        print(f"  Generation time: {result['generation_time']:.2f}s")
        print(f"  Total time: {result['total_time']:.2f}s")
        print(f"  Retries: {result['retries']}")
        
        if 'alignment_error' in result:
            print(f"  Note: Alignment failed - {result['alignment_error']}")
        
        # Now visualize landmarks on both template and result
        aligner = FaceAligner()
        
        # Load images
        template_img = cv2.imread(str(template_path))
        result_img = cv2.imread(str(output_path))
        
        # Acid green color
        acid_green = (0, 255, 127)
        
        # Try to detect landmarks in template (might fail for cartoon)
        print("\nDetecting landmarks in template...")
        template_landmarks = aligner.detect_landmarks(template_path)
        
        if template_landmarks['success']:
            print("✓ Template landmarks detected")
            # Draw landmarks
            for point in template_landmarks['landmarks']:
                x, y = int(point[0]), int(point[1])
                cv2.circle(template_img, (x, y), 5, acid_green, -1)
                cv2.circle(template_img, (x, y), 7, acid_green, 2)
            
            # Draw eye center
            eye_center = template_landmarks['eye_center']
            x, y = int(eye_center[0]), int(eye_center[1])
            cv2.circle(template_img, (x, y), 10, acid_green, -1)
            cv2.circle(template_img, (x, y), 12, acid_green, 2)
            cv2.putText(template_img, 'Eyes', (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, acid_green, 2)
            
            # Draw mouth center
            mouth_center = template_landmarks['mouth_center']
            x, y = int(mouth_center[0]), int(mouth_center[1])
            cv2.circle(template_img, (x, y), 10, acid_green, -1)
            cv2.circle(template_img, (x, y), 12, acid_green, 2)
            cv2.putText(template_img, 'Mouth', (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, acid_green, 2)
            
            # Draw line
            cv2.line(template_img,
                     (int(eye_center[0]), int(eye_center[1])),
                     (int(mouth_center[0]), int(mouth_center[1])),
                     acid_green, 3)
        else:
            print(f"✗ Template landmarks not detected: {template_landmarks['error']}")
            # Draw manual landmarks for visualization
            h, w = template_img.shape[:2]
            eye_center = np.array([w * 0.5, h * 0.4])
            mouth_center = np.array([w * 0.5, h * 0.65])
            
            x, y = int(eye_center[0]), int(eye_center[1])
            cv2.circle(template_img, (x, y), 10, acid_green, -1)
            cv2.putText(template_img, 'Eyes (approx)', (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, acid_green, 2)
            
            x, y = int(mouth_center[0]), int(mouth_center[1])
            cv2.circle(template_img, (x, y), 10, acid_green, -1)
            cv2.putText(template_img, 'Mouth (approx)', (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, acid_green, 2)
            
            cv2.line(template_img,
                     (int(eye_center[0]), int(eye_center[1])),
                     (int(mouth_center[0]), int(mouth_center[1])),
                     acid_green, 3)
        
        # Detect landmarks in result
        print("\nDetecting landmarks in result...")
        result_landmarks = aligner.detect_landmarks(output_path)
        
        if result_landmarks['success']:
            print("✓ Result landmarks detected")
            # Draw landmarks
            for point in result_landmarks['landmarks']:
                x, y = int(point[0]), int(point[1])
                cv2.circle(result_img, (x, y), 5, acid_green, -1)
                cv2.circle(result_img, (x, y), 7, acid_green, 2)
            
            # Draw eye center
            eye_center = result_landmarks['eye_center']
            x, y = int(eye_center[0]), int(eye_center[1])
            cv2.circle(result_img, (x, y), 10, acid_green, -1)
            cv2.circle(result_img, (x, y), 12, acid_green, 2)
            cv2.putText(result_img, 'Eyes', (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, acid_green, 2)
            
            # Draw mouth center
            mouth_center = result_landmarks['mouth_center']
            x, y = int(mouth_center[0]), int(mouth_center[1])
            cv2.circle(result_img, (x, y), 10, acid_green, -1)
            cv2.circle(result_img, (x, y), 12, acid_green, 2)
            cv2.putText(result_img, 'Mouth', (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, acid_green, 2)
            
            # Draw line
            cv2.line(result_img,
                     (int(eye_center[0]), int(eye_center[1])),
                     (int(mouth_center[0]), int(mouth_center[1])),
                     acid_green, 3)
            
            # Calculate distance
            distance = np.linalg.norm(mouth_center - eye_center)
            mid_x = int((eye_center[0] + mouth_center[0]) / 2)
            mid_y = int((eye_center[1] + mouth_center[1]) / 2)
            cv2.putText(result_img, f'd={distance:.1f}px', (mid_x+10, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, acid_green, 2)
        else:
            print(f"✗ Result landmarks not detected: {result_landmarks['error']}")
        
        # Save visualizations
        template_vis_path = Path("test_template_with_landmarks.png")
        result_vis_path = Path("test_result_with_landmarks.png")
        
        cv2.imwrite(str(template_vis_path), template_img)
        cv2.imwrite(str(result_vis_path), result_img)
        
        print(f"\nSaved visualizations:")
        print(f"- Template with landmarks: {template_vis_path}")
        print(f"- Result with landmarks: {result_vis_path}")
        
        # Create side-by-side comparison
        # Resize to same height for comparison
        h1, w1 = template_img.shape[:2]
        h2, w2 = result_img.shape[:2]
        
        if h1 != h2:
            # Scale result to match template height
            scale = h1 / h2
            new_w2 = int(w2 * scale)
            result_img_resized = cv2.resize(result_img, (new_w2, h1))
        else:
            result_img_resized = result_img
        
        # Create comparison
        comparison = np.hstack([template_img, result_img_resized])
        comparison_path = Path("test_input_output_comparison.png")
        cv2.imwrite(str(comparison_path), comparison)
        print(f"- Side-by-side comparison: {comparison_path}")
        
    else:
        print(f"✗ Face swap failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_single_pair()