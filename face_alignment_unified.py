#!/usr/bin/env python3
"""
Unified face alignment using InsightFace first, then MediaPipe as fallback.
Detectors only provide keypoints; aligner computes the transform.
Thread-safe by avoiding per-call model reconfiguration and avoiding shared mutable state.
"""
import numpy as np
import cv2
from pathlib import Path
import logging
import threading

# Third-party detectors
from insightface.app import FaceAnalysis
import mediapipe as mp


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsightFaceDetector:
    """Wraps InsightFace landmark detection with fixed threshold post-init."""
    def __init__(self, det_size=(640, 640), det_thresh=0.35):
        self._lock = threading.Lock()
        try:
            app = FaceAnalysis(
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection', 'landmark_2d_106']
            )
            app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
            self._app = app
            self._ok = True
            self._det_size = det_size
            self._det_thresh = det_thresh
        except Exception as e:
            logger.warning(f"InsightFace init failed: {e}")
            self._app = None
            self._ok = False

    def detect_landmarks(self, image_path):
        if not self._ok:
            return {'success': False, 'error': 'InsightFace not available'}
        img = cv2.imread(str(image_path))
        if img is None:
            return {'success': False, 'error': f'Failed to read image: {image_path}'}
        try:
            with self._lock:
                faces = self._app.get(img)
            if len(faces) == 0:
                return {'success': False, 'error': 'No face detected (InsightFace)'}
            face = faces[0]
            landmarks = face.kps
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]
            eye_center = (left_eye + right_eye) / 2
            mouth_center = (left_mouth + right_mouth) / 2
            return {
                'success': True,
                'landmarks': landmarks,
                'eye_center': eye_center,
                'mouth_center': mouth_center,
                'bbox': face.bbox,
                'method': 'insightface',
                'error': None
            }
        except Exception as e:
            return {'success': False, 'error': f'InsightFace detection failed: {str(e)}'}


class MediaPipeDetector:
    """Wraps MediaPipe FaceMesh for landmark detection on static images."""
    def __init__(self):
        self._lock = threading.Lock()
        self._mp = mp
        try:
            self._face_mesh = self._mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5
            )
            self._ok = True
        except Exception as e:
            logger.warning(f"MediaPipe init failed: {e}")
            self._face_mesh = None
            self._ok = False

        # Predefined landmark index sets (FaceMesh 468 points)
        # Use robust sets to compute centers by averaging
        self._left_eye_idx = [33, 133, 160, 159, 158, 157, 173]
        self._right_eye_idx = [263, 362, 387, 386, 385, 384, 398]
        # Mouth: use upper (13) and lower (14) for center, plus corners for bbox
        self._mouth_center_idx = [13, 14]
        self._mouth_corner_idx = [61, 291]

    def _landmark_xy(self, lm, w, h):
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)

    def detect_landmarks(self, image_path):
        if not self._ok:
            return {'success': False, 'error': 'MediaPipe not available'}
        img = cv2.imread(str(image_path))
        if img is None:
            return {'success': False, 'error': f'Failed to read image: {image_path}'}
        h, w = img.shape[:2]
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with self._lock:
                result = self._face_mesh.process(rgb)
            if not result.multi_face_landmarks:
                return {'success': False, 'error': 'No face detected (MediaPipe)'}

            face_lms = result.multi_face_landmarks[0].landmark

            left_eye_pts = np.array([self._landmark_xy(face_lms[i], w, h) for i in self._left_eye_idx], dtype=np.float32)
            right_eye_pts = np.array([self._landmark_xy(face_lms[i], w, h) for i in self._right_eye_idx], dtype=np.float32)
            eye_center = (left_eye_pts.mean(axis=0) + right_eye_pts.mean(axis=0)) / 2.0

            mouth_pts = np.array([self._landmark_xy(face_lms[i], w, h) for i in self._mouth_center_idx], dtype=np.float32)
            mouth_center = mouth_pts.mean(axis=0)

            # Build a minimal 5-point landmark array for compatibility
            # left_eye, right_eye, nose (approx as midpoint of eyes), left_mouth, right_mouth
            left_eye_center = left_eye_pts.mean(axis=0)
            right_eye_center = right_eye_pts.mean(axis=0)
            left_mouth = self._landmark_xy(face_lms[self._mouth_corner_idx[0]], w, h)
            right_mouth = self._landmark_xy(face_lms[self._mouth_corner_idx[1]], w, h)
            nose_approx = (left_eye_center + right_eye_center) / 2.0
            landmarks = np.stack([left_eye_center, right_eye_center, nose_approx, left_mouth, right_mouth]).astype(np.float32)

            # Compute bbox from all used points
            all_pts = np.vstack([left_eye_pts, right_eye_pts, mouth_pts, [left_mouth, right_mouth]])
            x_min, y_min = all_pts.min(axis=0)
            x_max, y_max = all_pts.max(axis=0)
            bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

            return {
                'success': True,
                'landmarks': landmarks,
                'eye_center': eye_center,
                'mouth_center': mouth_center,
                'bbox': bbox,
                'method': 'mediapipe',
                'error': None
            }
        except Exception as e:
            return {'success': False, 'error': f'MediaPipe detection failed: {str(e)}'}


class UnifiedFaceAligner:
    """Aligner that queries InsightFace first, then MediaPipe for landmarks."""
    def __init__(self):
        self._insight = InsightFaceDetector()
        self._mediapipe = MediaPipeDetector()

    def detect_landmarks(self, image_path):
        # Try InsightFace first
        result = self._insight.detect_landmarks(image_path)
        if result.get('success'):
            return result
        # Fallback to MediaPipe
        result = self._mediapipe.detect_landmarks(image_path)
        return result

    @staticmethod
    def _calculate_alignment_params(template_landmarks, generated_landmarks):
        t_eye = template_landmarks['eye_center']
        t_mouth = template_landmarks['mouth_center']
        g_eye = generated_landmarks['eye_center']
        g_mouth = generated_landmarks['mouth_center']

        t_vec = t_mouth - t_eye
        g_vec = g_mouth - g_eye

        t_dist = np.linalg.norm(t_vec)
        g_dist = np.linalg.norm(g_vec)
        scale = t_dist / g_dist if g_dist > 0 else 1.0

        t_angle = np.arctan2(t_vec[1], t_vec[0])
        g_angle = np.arctan2(g_vec[1], g_vec[0])
        angle_deg = np.degrees(t_angle - g_angle)

        return {
            'scale': float(scale),
            'angle': float(angle_deg),
            'template_eye_center': t_eye,
            'generated_eye_center': g_eye,
        }

    def align_image(self, generated_image_path, template_image_path, output_path=None):
        if output_path is None:
            output_path = generated_image_path

        # Detect landmarks
        template_result = self.detect_landmarks(template_image_path)
        if not template_result.get('success'):
            return {'success': False, 'error': f"Template detection failed: {template_result.get('error', 'unknown')}", 'retries': 0}

        generated_result = self.detect_landmarks(generated_image_path)
        if not generated_result.get('success'):
            return {'success': False, 'error': f"Generated detection failed: {generated_result.get('error', 'unknown')}", 'retries': 0}

        try:
            gen_img = cv2.imread(str(generated_image_path))
            tmpl_img = cv2.imread(str(template_image_path))
            if gen_img is None or tmpl_img is None:
                return {'success': False, 'error': 'Failed to read images'}

            h_t, w_t = tmpl_img.shape[:2]
            params = self._calculate_alignment_params(template_result, generated_result)

            gen_eye = params['generated_eye_center']
            tmpl_eye = params['template_eye_center']

            T1 = np.array([[1, 0, -gen_eye[0]],
                           [0, 1, -gen_eye[1]],
                           [0, 0, 1]], dtype=np.float32)
            angle_rad = np.radians(params['angle'])
            c = np.cos(angle_rad) * params['scale']
            s = np.sin(angle_rad) * params['scale']
            R = np.array([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]], dtype=np.float32)
            T2 = np.array([[1, 0, tmpl_eye[0]],
                           [0, 1, tmpl_eye[1]],
                           [0, 0, 1]], dtype=np.float32)
            M = T2 @ R @ T1
            M_2x3 = M[:2, :]

            aligned = cv2.warpAffine(gen_img, M_2x3, (w_t, h_t))
            cv2.imwrite(str(output_path), aligned)
            return {
                'success': True,
                'output_path': str(output_path),
                'scale': params['scale'],
                'angle': params['angle'],
                'template_method': template_result.get('method', 'unknown'),
                'generated_method': generated_result.get('method', 'unknown')
            }
        except Exception as e:
            return {'success': False, 'error': f'Alignment failed: {str(e)}'}


def align_generated_image(generated_path, template_path, output_path=None):
    """Stateless convenience function; creates an aligner per call for thread-safety."""
    aligner = UnifiedFaceAligner()
    return aligner.align_image(generated_path, template_path, output_path)



