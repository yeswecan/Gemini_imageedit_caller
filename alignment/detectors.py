#!/usr/bin/env python3
import threading
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
from insightface.app import FaceAnalysis
import mediapipe as mp


@dataclass
class DetectedLandmarks:
    left_eye_center: np.ndarray
    right_eye_center: np.ndarray
    mouth_center: np.ndarray
    points5: np.ndarray
    method: str


class InsightFaceLandmarkDetector:
    def __init__(self, det_size=(640, 640), det_thresh=0.35):
        self._lock = threading.Lock()
        self._app = FaceAnalysis(
            providers=['CPUExecutionProvider'],
            allowed_modules=['detection', 'landmark_2d_106']
        )
        self._app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)

    def detect(self, image_path: Path):
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        with self._lock:
            faces = self._app.get(img)
        if len(faces) == 0:
            return None
        face = faces[0]
        kps = face.kps
        left_eye = kps[0]
        right_eye = kps[1]
        left_mouth = kps[3]
        right_mouth = kps[4]
        eye_center = (left_eye + right_eye) / 2
        mouth_center = (left_mouth + right_mouth) / 2
        points5 = np.stack([left_eye, right_eye, (left_eye + right_eye) / 2, left_mouth, right_mouth]).astype(np.float32)
        return DetectedLandmarks(
            left_eye_center=left_eye.astype(np.float32),
            right_eye_center=right_eye.astype(np.float32),
            mouth_center=mouth_center.astype(np.float32),
            points5=points5,
            method='insightface'
        )


class MediapipeLandmarkDetector:
    def __init__(self):
        self._mp = mp
        self._lock = threading.Lock()
        self._face_mesh = self._mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.3
        )
        self._left_eye_idx = [33, 133, 160, 159, 158, 157, 173, 246, 7, 163]
        self._right_eye_idx = [263, 362, 387, 386, 385, 384, 398, 466, 249, 390]
        self._mouth_center_idx = [13, 14, 0]
        self._mouth_corner_idx = [61, 291]

    def _xy(self, lm, w, h):
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)

    def detect(self, image_path: Path):
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self._lock:
            res = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lms = res.multi_face_landmarks[0].landmark
        left_eye_pts = np.array([self._xy(lms[i], w, h) for i in self._left_eye_idx])
        right_eye_pts = np.array([self._xy(lms[i], w, h) for i in self._right_eye_idx])
        left_eye_center = left_eye_pts.mean(axis=0)
        right_eye_center = right_eye_pts.mean(axis=0)
        mouth_pts = np.array([self._xy(lms[i], w, h) for i in self._mouth_center_idx])
        mouth_center = mouth_pts.mean(axis=0)
        left_mouth = self._xy(lms[self._mouth_corner_idx[0]], w, h)
        right_mouth = self._xy(lms[self._mouth_corner_idx[1]], w, h)
        points5 = np.stack([left_eye_center, right_eye_center, (left_eye_center + right_eye_center) / 2, left_mouth, right_mouth]).astype(np.float32)
        return DetectedLandmarks(
            left_eye_center=left_eye_center.astype(np.float32),
            right_eye_center=right_eye_center.astype(np.float32),
            mouth_center=mouth_center.astype(np.float32),
            points5=points5,
            method='mediapipe'
        )


class CombinedLandmarkDetector:
    def __init__(self):
        self.insight = InsightFaceLandmarkDetector()
        self.mediapipe = MediapipeLandmarkDetector()

    def detect_both_images(self, template_path: Path, generated_path: Path, bypass_insightface=False, bypass_mediapipe=False):
        # Try InsightFace for both
        ins_template = None if bypass_insightface else self.insight.detect(template_path)
        ins_generated = None if bypass_insightface else self.insight.detect(generated_path)
        if ins_template is not None and ins_generated is not None:
            return ins_template, ins_generated, 'insightface'

        # Try Mediapipe for both
        mp_template = None if bypass_mediapipe else self.mediapipe.detect(template_path)
        mp_generated = None if bypass_mediapipe else self.mediapipe.detect(generated_path)
        if mp_template is not None and mp_generated is not None:
            return mp_template, mp_generated, 'mediapipe'

        return None, None, 'none'


