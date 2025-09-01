#!/usr/bin/env python3
"""
Test face alignment with visual debugging - draws landmarks in acid green.
"""
import cv2
import numpy as np
from pathlib import Path
from face_alignment import FaceAligner
import sys

def visualize_landmarks(image_path, output_path):
    """
    Detect landmarks and draw them on the image.
    Returns the landmarks data for debugging.
    """
    # Initialize aligner
    aligner = FaceAligner()
    
    # Detect landmarks
    result = aligner.detect_landmarks(image_path)
    
    if not result['success']:
        print(f"Failed to detect landmarks: {result['error']}")
        return None
    
    # Read image
    img = cv2.imread(str(image_path))
    
    # Acid green color (BGR format)
    acid_green = (0, 255, 127)
    
    # Draw all 5 landmarks
    landmarks = result['landmarks']
    for i, point in enumerate(landmarks):
        x, y = int(point[0]), int(point[1])
        # Draw circle
        cv2.circle(img, (x, y), 5, acid_green, -1)
        cv2.circle(img, (x, y), 7, acid_green, 2)
        
        # Label points
        labels = ['L_Eye', 'R_Eye', 'Nose', 'L_Mouth', 'R_Mouth']
        cv2.putText(img, labels[i], (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, acid_green, 2)
    
    # Draw eye center
    eye_center = result['eye_center']
    x, y = int(eye_center[0]), int(eye_center[1])
    cv2.circle(img, (x, y), 8, acid_green, -1)
    cv2.circle(img, (x, y), 10, acid_green, 2)
    cv2.putText(img, 'Eye_Center', (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, acid_green, 2)
    
    # Draw mouth center
    mouth_center = result['mouth_center']
    x, y = int(mouth_center[0]), int(mouth_center[1])
    cv2.circle(img, (x, y), 8, acid_green, -1)
    cv2.circle(img, (x, y), 10, acid_green, 2)
    cv2.putText(img, 'Mouth_Center', (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, acid_green, 2)
    
    # Draw line between eye center and mouth center
    cv2.line(img, 
             (int(eye_center[0]), int(eye_center[1])),
             (int(mouth_center[0]), int(mouth_center[1])),
             acid_green, 2)
    
    # Calculate and display distance
    distance = np.linalg.norm(mouth_center - eye_center)
    mid_x = int((eye_center[0] + mouth_center[0]) / 2)
    mid_y = int((eye_center[1] + mouth_center[1]) / 2)
    cv2.putText(img, f'd={distance:.1f}', (mid_x+10, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, acid_green, 2)
    
    # Save visualized image
    cv2.imwrite(str(output_path), img)
    print(f"Saved visualization to: {output_path}")
    
    return result

def test_single_alignment():
    """Test alignment on a single image pair with visualization."""
    # Use existing result for testing
    template_path = Path("characters/Ali_F.png")
    generated_path = Path("results/Ali_F_005_Selfie_12_result.png")
    
    if not generated_path.exists():
        print(f"Generated image not found: {generated_path}")
        print("Please run generate_all_results.py first to create test images.")
        return
    
    print("Testing face alignment with visualization...")
    print(f"Template: {template_path}")
    print(f"Generated: {generated_path}")
    print()
    
    # Visualize landmarks on template
    print("1. Detecting landmarks in template...")
    template_vis_path = Path("test_template_landmarks.png")
    template_result = visualize_landmarks(template_path, template_vis_path)
    
    if template_result:
        print(f"   Template eye center: {template_result['eye_center']}")
        print(f"   Template mouth center: {template_result['mouth_center']}")
    
    # Visualize landmarks on generated image
    print("\n2. Detecting landmarks in generated image...")
    generated_vis_path = Path("test_generated_landmarks.png")
    generated_result = visualize_landmarks(generated_path, generated_vis_path)
    
    if generated_result:
        print(f"   Generated eye center: {generated_result['eye_center']}")
        print(f"   Generated mouth center: {generated_result['mouth_center']}")
    
    # Test alignment
    aligned_path = Path("test_aligned_result.png")
    aligned_vis_path = Path("test_aligned_landmarks.png")
    
    if template_result and generated_result:
        print("\n3. Testing alignment...")
        aligner = FaceAligner()
        
        alignment_result = aligner.align_image(
            generated_path,
            template_path,
            aligned_path
        )
        
        if alignment_result['success']:
            print(f"   Alignment successful!")
            print(f"   Scale factor: {alignment_result['scale']:.3f}")
            print(f"   Rotation angle: {alignment_result['angle']:.1f} degrees")
            
            # Visualize landmarks on aligned result
            print("\n4. Verifying aligned result...")
            aligned_result = visualize_landmarks(aligned_path, aligned_vis_path)
            
            if aligned_result:
                print(f"   Aligned eye center: {aligned_result['eye_center']}")
                print(f"   Aligned mouth center: {aligned_result['mouth_center']}")
                
                # Check if alignment worked
                template_eye_dist = np.linalg.norm(template_result['mouth_center'] - template_result['eye_center'])
                aligned_eye_dist = np.linalg.norm(aligned_result['mouth_center'] - aligned_result['eye_center'])
                
                print(f"\n   Distance comparison:")
                print(f"   Template eye-mouth distance: {template_eye_dist:.1f}")
                print(f"   Aligned eye-mouth distance: {aligned_eye_dist:.1f}")
                print(f"   Difference: {abs(template_eye_dist - aligned_eye_dist):.1f} pixels")
        else:
            print(f"   Alignment failed: {alignment_result['error']}")
    else:
        print("\n3. Cannot test alignment - landmarks not detected")
        print("   This is expected for stylized illustrations.")
        print("   InsightFace works best with photorealistic faces.")
    
    print("\nVisualization files created:")
    if template_vis_path.exists():
        print(f"- {template_vis_path} (template with landmarks)")
    if generated_vis_path.exists():
        print(f"- {generated_vis_path} (generated with landmarks)")
    if aligned_path.exists() and aligned_vis_path.exists():
        print(f"- {aligned_vis_path} (aligned with landmarks)")

if __name__ == "__main__":
    test_single_alignment()