#!/usr/bin/env python3
"""
Test InsightFace on selfie photos to verify it's working.
"""
import cv2
import numpy as np
from pathlib import Path
from face_alignment import FaceAligner

def test_on_selfie():
    """Test landmark detection on a real selfie photo."""
    print("Testing InsightFace on selfie photos...")
    
    # Test on a selfie
    selfie_path = Path("selfies_samples/005_Selfie_12.jpg")
    
    if not selfie_path.exists():
        print(f"Selfie not found: {selfie_path}")
        return
    
    print(f"\nTesting on: {selfie_path}")
    
    # Initialize aligner
    aligner = FaceAligner()
    
    # Detect landmarks
    result = aligner.detect_landmarks(selfie_path)
    
    if result['success']:
        print("✓ Successfully detected face landmarks!")
        print(f"  Eye center: {result['eye_center']}")
        print(f"  Mouth center: {result['mouth_center']}")
        print(f"  Landmarks shape: {result['landmarks'].shape}")
        
        # Visualize
        img = cv2.imread(str(selfie_path))
        
        # Acid green color
        acid_green = (0, 255, 127)
        
        # Draw all 5 landmarks
        landmarks = result['landmarks']
        for i, point in enumerate(landmarks):
            x, y = int(point[0]), int(point[1])
            cv2.circle(img, (x, y), 5, acid_green, -1)
            cv2.circle(img, (x, y), 7, acid_green, 2)
        
        # Draw eye center
        eye_center = result['eye_center']
        x, y = int(eye_center[0]), int(eye_center[1])
        cv2.circle(img, (x, y), 8, acid_green, -1)
        cv2.circle(img, (x, y), 10, acid_green, 2)
        
        # Draw mouth center
        mouth_center = result['mouth_center']
        x, y = int(mouth_center[0]), int(mouth_center[1])
        cv2.circle(img, (x, y), 8, acid_green, -1)
        cv2.circle(img, (x, y), 10, acid_green, 2)
        
        # Draw line
        cv2.line(img, 
                 (int(eye_center[0]), int(eye_center[1])),
                 (int(mouth_center[0]), int(mouth_center[1])),
                 acid_green, 2)
        
        # Save
        output_path = Path("test_selfie_landmarks.png")
        cv2.imwrite(str(output_path), img)
        print(f"\nSaved visualization to: {output_path}")
        
        # Now test on a generated result that should have a more realistic face
        print("\n\nTesting on generated result...")
        generated_path = Path("results/Ali_F_005_Selfie_12_result.png")
        if generated_path.exists():
            gen_result = aligner.detect_landmarks(generated_path)
            if gen_result['success']:
                print("✓ Successfully detected landmarks in generated image!")
                print(f"  Eye center: {gen_result['eye_center']}")
                print(f"  Mouth center: {gen_result['mouth_center']}")
                
                # Visualize generated
                img_gen = cv2.imread(str(generated_path))
                for point in gen_result['landmarks']:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(img_gen, (x, y), 5, acid_green, -1)
                    cv2.circle(img_gen, (x, y), 7, acid_green, 2)
                
                output_gen = Path("test_generated_landmarks_real.png")
                cv2.imwrite(str(output_gen), img_gen)
                print(f"  Saved to: {output_gen}")
            else:
                print(f"✗ Failed on generated: {gen_result['error']}")
    else:
        print(f"✗ Failed to detect landmarks: {result['error']}")

if __name__ == "__main__":
    test_on_selfie()