#!/usr/bin/env python3
from pathlib import Path
from typing import Tuple
import numpy as np
import cv2

from alignment.detectors import CombinedLandmarkDetector, DetectedLandmarks, InsightFaceLandmarkDetector, MediapipeLandmarkDetector
from alignment.transform import compute_similarity_transform_from_three_points, warp_generated_to_template_canvas


def _to_triplet_points(lm: DetectedLandmarks) -> np.ndarray:
    return np.stack([lm.left_eye_center, lm.right_eye_center, lm.mouth_center], axis=0).astype(np.float32)


def select_consistent_landmarks(
    template_path: Path,
    generated_path: Path,
    disable_insightface: bool = False,
    disable_mediapipe: bool = False,
) -> Tuple[DetectedLandmarks, DetectedLandmarks, str]:
    detector = CombinedLandmarkDetector()
    tmpl, gen, method = detector.detect_both_images(
        template_path=template_path,
        generated_path=generated_path,
        bypass_insightface=disable_insightface,
        bypass_mediapipe=disable_mediapipe,
    )
    if tmpl is None or gen is None or method == 'none':
        raise ValueError('Landmark detection failed consistently for both images')
    return tmpl, gen, method


def align_generated_to_template(
    generated_path: Path,
    template_path: Path,
    output_path: Path,
    disable_insightface: bool = False,
    disable_mediapipe: bool = False,
) -> dict:
    # Read template to get size (width,height)
    tmpl_img = cv2.imread(str(template_path))
    if tmpl_img is None:
        return { 'success': False, 'error': f'Failed to read template: {template_path}'}
    h_t, w_t = tmpl_img.shape[:2]

    try:
        tmpl_lm, gen_lm, method = select_consistent_landmarks(
            template_path, generated_path,
            disable_insightface=disable_insightface,
            disable_mediapipe=disable_mediapipe,
        )
    except Exception as e:
        return { 'success': False, 'error': str(e) }

    tpl_pts = _to_triplet_points(tmpl_lm)
    gen_pts = _to_triplet_points(gen_lm)

    M = compute_similarity_transform_from_three_points(tpl_pts, gen_pts)
    if M is None:
        return { 'success': False, 'error': 'Failed to compute similarity transform' }

    ok = warp_generated_to_template_canvas(
        generated_path=generated_path,
        template_size=(w_t, h_t),
        affine_2x3=M,
        output_path=output_path,
    )
    if not ok:
        return { 'success': False, 'error': 'Warp failed' }

    return {
        'success': True,
        'method': method,
        'affine': M.tolist(),
        'template_size': [w_t, h_t],
        'output_path': str(output_path),
    }



