#!/usr/bin/env python3
"""
Face alignment module using InsightFace for landmark detection.
Aligns generated faces to match the template illustration's face position and scale.
"""
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceAligner:
    def __init__(self, det_size=(640, 640)):
        """Initialize face analysis app with InsightFace"""
        # Use a more permissive configuration for illustrations
        self.app = FaceAnalysis(
            providers=['CPUExecutionProvider'],
            allowed_modules=['detection', 'landmark_2d_106']  # Focus on 2D landmarks
        )
        self.app.prepare(ctx_id=0, det_size=det_size, det_thresh=0.3)  # Lower threshold for illustrations
        
    def detect_landmarks(self, image_path):
        """
        Detect face landmarks in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict with 'success', 'landmarks', 'bbox', and 'error' keys
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return {'success': False, 'error': f'Failed to read image: {image_path}'}
            
            # Detect faces
            faces = self.app.get(img)
            
            if len(faces) == 0:
                return {'success': False, 'error': 'No face detected in image'}
            
            # Use the first detected face
            face = faces[0]
            
            # Extract landmarks (InsightFace provides 5 key points by default)
            # Points are: left eye, right eye, nose tip, left mouth corner, right mouth corner
            landmarks = face.kps
            
            # Calculate center points
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]
            
            # Center between eyes
            eye_center = (left_eye + right_eye) / 2
            
            # Center of mouth
            mouth_center = (left_mouth + right_mouth) / 2
            
            return {
                'success': True,
                'landmarks': landmarks,
                'eye_center': eye_center,
                'mouth_center': mouth_center,
                'bbox': face.bbox,
                'error': None
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Landmark detection failed: {str(e)}'}
    
    def calculate_alignment_params(self, template_landmarks, generated_landmarks):
        """
        Calculate scale, rotation, and translation to align generated face to template.
        
        Args:
            template_landmarks: Landmarks from template illustration
            generated_landmarks: Landmarks from generated image
            
        Returns:
            dict with 'scale', 'angle', 'translation' keys
        """
        # Extract key points
        template_eye_center = template_landmarks['eye_center']
        template_mouth_center = template_landmarks['mouth_center']
        generated_eye_center = generated_landmarks['eye_center']
        generated_mouth_center = generated_landmarks['mouth_center']
        
        # Calculate distances (eye center to mouth center)
        template_distance = np.linalg.norm(template_mouth_center - template_eye_center)
        generated_distance = np.linalg.norm(generated_mouth_center - generated_eye_center)
        
        # Calculate scale factor
        scale = template_distance / generated_distance if generated_distance > 0 else 1.0
        
        # Calculate rotation angle
        template_vector = template_mouth_center - template_eye_center
        generated_vector = generated_mouth_center - generated_eye_center
        
        template_angle = np.arctan2(template_vector[1], template_vector[0])
        generated_angle = np.arctan2(generated_vector[1], generated_vector[0])
        
        angle = np.degrees(template_angle - generated_angle)
        
        # Calculate translation (after scaling and rotation)
        # This will be refined after transformation
        translation = template_eye_center - generated_eye_center
        
        return {
            'scale': scale,
            'angle': angle,
            'translation': translation,
            'template_eye_center': template_eye_center,
            'template_mouth_center': template_mouth_center
        }
    
    def align_image(self, generated_image_path, template_image_path, output_path=None):
        """
        Align generated image to match template face position and scale.
        
        Args:
            generated_image_path: Path to generated image
            template_image_path: Path to template illustration
            output_path: Optional output path (defaults to overwriting generated image)
            
        Returns:
            dict with 'success', 'output_path', 'retries', and 'error' keys
        """
        if output_path is None:
            output_path = generated_image_path
            
        retries = 0
        
        # Detect landmarks in template
        template_result = self.detect_landmarks(template_image_path)
        if not template_result['success']:
            return {
                'success': False,
                'error': f"Template landmark detection failed: {template_result['error']}",
                'retries': retries
            }
        
        # Detect landmarks in generated image
        generated_result = self.detect_landmarks(generated_image_path)
        if not generated_result['success']:
            return {
                'success': False,
                'error': f"Generated image landmark detection failed: {generated_result['error']}",
                'retries': retries
            }
        
        try:
            # Read images
            generated_img = cv2.imread(str(generated_image_path))
            template_img = cv2.imread(str(template_image_path))
            
            # Calculate alignment parameters
            align_params = self.calculate_alignment_params(template_result, generated_result)
            
            # Get image dimensions
            h, w = template_img.shape[:2]
            
            # Create transformation matrix
            # First, scale and rotate around the generated eye center
            eye_center = generated_result['eye_center']
            center_point = (int(eye_center[0]), int(eye_center[1]))
            M1 = cv2.getRotationMatrix2D(
                center_point,
                align_params['angle'],
                align_params['scale']
            )
            
            # Apply initial transformation
            transformed = cv2.warpAffine(generated_img, M1, (w, h))
            
            # Detect landmarks again in transformed image to refine translation
            temp_path = Path(output_path).parent / "temp_aligned.png"
            cv2.imwrite(str(temp_path), transformed)
            
            transformed_result = self.detect_landmarks(temp_path)
            if transformed_result['success']:
                # Calculate final translation
                final_translation = align_params['template_eye_center'] - transformed_result['eye_center']
                
                # Apply translation
                M2 = np.float32([[1, 0, final_translation[0]], [0, 1, final_translation[1]]])
                aligned = cv2.warpAffine(transformed, M2, (w, h))
            else:
                # Fallback: use the initial transformation
                aligned = transformed
                
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            
            # Save aligned image
            cv2.imwrite(str(output_path), aligned)
            
            return {
                'success': True,
                'output_path': str(output_path),
                'retries': retries,
                'scale': align_params['scale'],
                'angle': align_params['angle']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Alignment failed: {str(e)}",
                'retries': retries
            }

# Singleton instance
_aligner = None

def get_aligner():
    """Get or create the singleton FaceAligner instance"""
    global _aligner
    if _aligner is None:
        _aligner = FaceAligner()
    return _aligner

def align_generated_image(generated_path, template_path, output_path=None):
    """
    Convenience function to align a generated image to a template.
    
    Args:
        generated_path: Path to generated image
        template_path: Path to template illustration
        output_path: Optional output path
        
    Returns:
        dict with alignment results
    """
    aligner = get_aligner()
    return aligner.align_image(generated_path, template_path, output_path)