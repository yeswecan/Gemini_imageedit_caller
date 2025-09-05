#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import cv2


def compute_similarity_transform_from_three_points(template_points: np.ndarray, generated_points: np.ndarray) -> np.ndarray:
    """
    Compute 2x3 affine (similarity) transform that maps generated_points -> template_points.
    Each input is shape (3,2): [left_eye, right_eye, mouth].
    """
    assert template_points.shape == (3, 2)
    assert generated_points.shape == (3, 2)
    # Use estimateAffinePartial2D for similarity-like transform (no shear)
    M, _ = cv2.estimateAffinePartial2D(generated_points.astype(np.float32), template_points.astype(np.float32), method=cv2.LMEDS)
    return M  # shape (2,3)


def warp_generated_to_template_canvas(generated_path: Path, template_size: tuple[int, int], affine_2x3: np.ndarray, output_path: Path) -> bool:
    """
    Warp generated image into a canvas of template_size using affine.
    template_size is (width, height).
    """
    gen_img = cv2.imread(str(generated_path))
    if gen_img is None or affine_2x3 is None:
        return False
    w_t, h_t = template_size
    aligned = cv2.warpAffine(gen_img, affine_2x3, (w_t, h_t))
    cv2.imwrite(str(output_path), aligned)
    return True


