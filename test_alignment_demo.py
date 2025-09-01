#!/usr/bin/env python3
"""
Test alignment between template and generated image.
"""
import cv2
import numpy as np
from pathlib import Path
from face_alignment import FaceAligner

def test_alignment():
    """Test alignment with template and generated images."""
    print("Testing face alignment...")
    
    # Use the cartoon template and generated result
    template_path = Path("characters/Ali_F.png")
    generated_path = Path("results/Ali_F_005_Selfie_12_result.png")
    
    if not template_path.exists() or not generated_path.exists():
        print("Images not found")
        return
    
    # Initialize aligner
    aligner = FaceAligner()
    
    # Since InsightFace doesn't work well on cartoon templates,
    # let's manually define approximate landmark positions for the template
    template_img = cv2.imread(str(template_path))
    h, w = template_img.shape[:2]
    
    # Approximate landmarks for cartoon (these would normally come from detection)
    # These are rough estimates based on typical cartoon face proportions
    template_eye_center = np.array([w * 0.5, h * 0.4])
    template_mouth_center = np.array([w * 0.5, h * 0.65])
    
    print(f"Template size: {w}x{h}")
    print(f"Template eye center (approx): {template_eye_center}")
    print(f"Template mouth center (approx): {template_mouth_center}")
    
    # Detect landmarks in generated image
    gen_result = aligner.detect_landmarks(generated_path)
    
    if gen_result['success']:
        print(f"\nGenerated landmarks detected!")
        print(f"Generated eye center: {gen_result['eye_center']}")
        print(f"Generated mouth center: {gen_result['mouth_center']}")
        
        # Calculate alignment parameters manually
        gen_eye_center = gen_result['eye_center']
        gen_mouth_center = gen_result['mouth_center']
        
        # Calculate distances
        template_distance = np.linalg.norm(template_mouth_center - template_eye_center)
        generated_distance = np.linalg.norm(gen_mouth_center - gen_eye_center)
        
        # Calculate scale
        scale = template_distance / generated_distance if generated_distance > 0 else 1.0
        
        # Calculate angle
        template_vector = template_mouth_center - template_eye_center
        generated_vector = gen_mouth_center - gen_eye_center
        
        template_angle = np.arctan2(template_vector[1], template_vector[0])
        generated_angle = np.arctan2(generated_vector[1], generated_vector[0])
        
        angle = np.degrees(template_angle - generated_angle)
        
        print(f"\nAlignment parameters:")
        print(f"Scale factor: {scale:.3f}")
        print(f"Rotation angle: {angle:.1f} degrees")
        print(f"Template face size: {template_distance:.1f} pixels")
        print(f"Generated face size: {generated_distance:.1f} pixels")
        
        # Apply transformation
        generated_img = cv2.imread(str(generated_path))
        
        # Create transformation matrix
        center_point = (int(gen_eye_center[0]), int(gen_eye_center[1]))
        M = cv2.getRotationMatrix2D(center_point, angle, scale)
        
        # Apply transformation
        aligned = cv2.warpAffine(generated_img, M, (w, h))
        
        # Now we need to detect where the face ended up and translate it to match template
        # Calculate where the eye center moved to after rotation/scale
        transformed_eye = M @ np.array([gen_eye_center[0], gen_eye_center[1], 1])
        
        # Calculate translation needed
        translation = template_eye_center - transformed_eye
        
        # Apply translation
        M2 = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
        aligned = cv2.warpAffine(aligned, M2, (w, h))
        
        # Save aligned result
        aligned_path = Path("test_alignment_demo_result.png")
        cv2.imwrite(str(aligned_path), aligned)
        print(f"\nSaved aligned image to: {aligned_path}")
        
        # Create comparison image
        comparison = np.hstack([template_img, generated_img, aligned])
        comp_path = Path("test_alignment_comparison.png")
        cv2.imwrite(str(comp_path), comparison)
        print(f"Saved comparison to: {comp_path}")
        
        # Visualize landmarks on all three
        acid_green = (0, 255, 127)
        
        # Template with manual landmarks
        template_vis = template_img.copy()
        cv2.circle(template_vis, tuple(template_eye_center.astype(int)), 10, acid_green, -1)
        cv2.circle(template_vis, tuple(template_mouth_center.astype(int)), 10, acid_green, -1)
        cv2.line(template_vis, 
                 tuple(template_eye_center.astype(int)),
                 tuple(template_mouth_center.astype(int)),
                 acid_green, 3)
        
        # Generated with detected landmarks
        generated_vis = generated_img.copy()
        for point in gen_result['landmarks']:
            cv2.circle(generated_vis, tuple(point.astype(int)), 5, acid_green, -1)
        cv2.circle(generated_vis, tuple(gen_eye_center.astype(int)), 10, acid_green, -1)
        cv2.circle(generated_vis, tuple(gen_mouth_center.astype(int)), 10, acid_green, -1)
        cv2.line(generated_vis,
                 tuple(gen_eye_center.astype(int)),
                 tuple(gen_mouth_center.astype(int)),
                 acid_green, 3)
        
        # Try to detect in aligned (might work since it's scaled/rotated)
        aligned_vis = aligned.copy()
        # Just copy the manual template landmarks for visualization
        cv2.circle(aligned_vis, tuple(template_eye_center.astype(int)), 10, acid_green, -1)
        cv2.circle(aligned_vis, tuple(template_mouth_center.astype(int)), 10, acid_green, -1)
        cv2.line(aligned_vis,
                 tuple(template_eye_center.astype(int)),
                 tuple(template_mouth_center.astype(int)),
                 acid_green, 3)
        
        # Create landmark comparison
        landmark_comp = np.hstack([template_vis, generated_vis, aligned_vis])
        landmark_path = Path("test_alignment_landmarks_comparison.png")
        cv2.imwrite(str(landmark_path), landmark_comp)
        print(f"Saved landmark comparison to: {landmark_path}")
        
    else:
        print(f"Failed to detect landmarks in generated image: {gen_result['error']}")

if __name__ == "__main__":
    test_alignment()