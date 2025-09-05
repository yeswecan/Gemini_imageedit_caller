#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import cv2

from alignment.pipeline import align_generated_to_template
from alignment.detectors import InsightFaceLandmarkDetector, MediapipeLandmarkDetector
from image_processor import ImageProcessor


def measure_per_axis_deltas(a: np.ndarray, b: np.ndarray):
    dx = float(abs(a[0] - b[0]))
    dy = float(abs(a[1] - b[1]))
    return dx, dy


def detect_with_both(image_path: Path):
    ins = InsightFaceLandmarkDetector()
    mp = MediapipeLandmarkDetector()
    ins_lm = ins.detect(image_path)
    mp_lm = mp.detect(image_path)
    return ins_lm, mp_lm


def render_overlay(image_path: Path, lm, out_path: Path):
    img = cv2.imread(str(image_path))
    if img is None or lm is None:
        return False
    color = (0, 255, 127)
    # small points for points5
    for p in lm.points5:
        x, y = int(p[0]), int(p[1])
        cv2.circle(img, (x, y), 4, color, -1)
        cv2.circle(img, (x, y), 6, color, 2)
    # larger for centers
    for cpt in [lm.left_eye_center, lm.right_eye_center, lm.mouth_center]:
        x, y = int(cpt[0]), int(cpt[1])
        cv2.circle(img, (x, y), 8, color, -1)
        cv2.circle(img, (x, y), 10, color, 2)
    # eye-mouth lines
    ec = ((lm.left_eye_center + lm.right_eye_center) / 2).astype(int)
    mc = lm.mouth_center.astype(int)
    cv2.line(img, (int(ec[0]), int(ec[1])), (int(mc[0]), int(mc[1])), color, 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def run_single_pair_consistency_test(template_path: Path, selfie_path: Path, work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    # Produce generated image via style transfer (no alignment here)
    generated_path = work_dir / 'generated.png'
    if not generated_path.exists():
        runner = ImageProcessor()
        gen_res = runner.process_images(
            template_path=template_path,
            selfie_path=selfie_path,
            output_path=generated_path,
            align=False
        )
        if not gen_res.get('success'):
            raise RuntimeError(f"Style transfer failed: {gen_res.get('error')}")

    # Align forcing InsightFace
    out_ins = work_dir / 'aligned_insightface.png'
    r1 = align_generated_to_template(generated_path, template_path, out_ins, disable_mediapipe=True)
    # Align forcing Mediapipe
    out_mp = work_dir / 'aligned_mediapipe.png'
    r2 = align_generated_to_template(generated_path, template_path, out_mp, disable_insightface=True)

    results = {
        'align_insightface': r1,
        'align_mediapipe': r2,
    }

    # Detect landmarks with both detectors on template and both outputs
    ins_t, mp_t = detect_with_both(template_path)
    ins_ai, mp_ai = detect_with_both(out_ins)
    ins_am, mp_am = detect_with_both(out_mp)

    # Save overlays
    overlays = work_dir / 'overlays'
    render_overlay(template_path, ins_t, overlays / 'template_insightface.png')
    render_overlay(template_path, mp_t, overlays / 'template_mediapipe.png')
    render_overlay(out_ins, ins_ai, overlays / 'aligned_insightface_insightface.png')
    render_overlay(out_ins, mp_ai, overlays / 'aligned_insightface_mediapipe.png')
    render_overlay(out_mp, ins_am, overlays / 'aligned_mediapipe_insightface.png')
    render_overlay(out_mp, mp_am, overlays / 'aligned_mediapipe_mediapipe.png')

    # Build metric rows for L eye, R eye, mouth
    def rows(lmA, lmB):
        if lmA is None or lmB is None:
            return {'left_eye': None, 'right_eye': None, 'mouth': None}
        return {
            'left_eye': measure_per_axis_deltas(lmA.left_eye_center, lmB.left_eye_center),
            'right_eye': measure_per_axis_deltas(lmA.right_eye_center, lmB.right_eye_center),
            'mouth': measure_per_axis_deltas(lmA.mouth_center, lmB.mouth_center),
        }

    metrics = {
        'template_mediapipe_vs_insightface': rows(mp_t, ins_t),
        'aligned_insight_mediapipe_vs_insightface': rows(mp_ai, ins_ai),
        'aligned_mediapipe_mediapipe_vs_insightface': rows(mp_am, ins_am),
    }

    # Check pass criteria |dx|,|dy| <= 5 for all three points where available
    def passed(m):
        if m is None:
            return False
        ok = True
        for k in ['left_eye', 'right_eye', 'mouth']:
            if m[k] is None:
                ok = False
                continue
            dx, dy = m[k]
            if dx > 5 or dy > 5:
                ok = False
        return ok

    results['metrics'] = metrics
    results['pass_template'] = passed(metrics['template_mediapipe_vs_insightface'])
    results['pass_aligned_insight'] = passed(metrics['aligned_insight_mediapipe_vs_insightface'])
    results['pass_aligned_mediapipe'] = passed(metrics['aligned_mediapipe_mediapipe_vs_insightface'])

    (work_dir / 'consistency_results.json').write_text(json.dumps(results, indent=2))
    return results


def main():
    template = Path('characters/Ali_F.png')
    selfie = Path('selfies_samples/005_Selfie_12.jpg')
    work = Path('debug/consistency_run')
    res = run_single_pair_consistency_test(template, selfie, work)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()


