#!/usr/bin/env python3
from pathlib import Path
import cv2
import numpy as np
from alignment.pipeline import align_generated_to_template
from alignment.detectors import InsightFaceLandmarkDetector, MediapipeLandmarkDetector
from image_processor import ImageProcessor


def render_overlay(image_path: Path, lm, out_path: Path):
    img = cv2.imread(str(image_path))
    if img is None or lm is None:
        return None
    color = (0, 255, 127)
    for p in lm.points5:
        x, y = int(p[0]), int(p[1])
        cv2.circle(img, (x, y), 4, color, -1)
        cv2.circle(img, (x, y), 6, color, 2)
    for cpt in [lm.left_eye_center, lm.right_eye_center, lm.mouth_center]:
        x, y = int(cpt[0]), int(cpt[1])
        cv2.circle(img, (x, y), 8, color, -1)
        cv2.circle(img, (x, y), 10, color, 2)
    ec = ((lm.left_eye_center + lm.right_eye_center) / 2).astype(int)
    mc = lm.mouth_center.astype(int)
    cv2.line(img, (int(ec[0]), int(ec[1])), (int(mc[0]), int(mc[1])), color, 2)
    cv2.imwrite(str(out_path), img)
    return out_path


def build_composite(template_path: Path, selfie_path: Path, out_path: Path, detector: str):
    # Clean work dir
    work = out_path.parent
    work.mkdir(parents=True, exist_ok=True)

    # Generate image via style transfer
    generated_path = work / f'gen_{detector}.png'
    runner = ImageProcessor()
    gen_res = runner.process_images(template_path, selfie_path, generated_path, align=False)
    if not gen_res.get('success'):
        raise RuntimeError(f'Generation failed: {gen_res.get("error")}')

    # Align using requested detector
    if detector == 'insightface':
        aligned_path = work / f'aligned_{detector}.png'
        ar = align_generated_to_template(generated_path, template_path, aligned_path, disable_mediapipe=True)
        det_t = InsightFaceLandmarkDetector()
        det_a = InsightFaceLandmarkDetector()
    elif detector == 'mediapipe':
        aligned_path = work / f'aligned_{detector}.png'
        ar = align_generated_to_template(generated_path, template_path, aligned_path, disable_insightface=True)
        det_t = MediapipeLandmarkDetector()
        det_a = MediapipeLandmarkDetector()
    else:
        raise ValueError('detector must be insightface or mediapipe')

    if not ar.get('success'):
        raise RuntimeError(f'Alignment failed: {ar.get("error")}')

    # Detect for overlays
    tmpl_lm = det_t.detect(template_path)
    aligned_lm = det_a.detect(aligned_path)

    # Render overlays
    tmpl_overlay = work / f'template_{detector}_overlay.png'
    aligned_overlay = work / f'aligned_{detector}_overlay.png'
    render_overlay(template_path, tmpl_lm, tmpl_overlay)
    render_overlay(aligned_path, aligned_lm, aligned_overlay)

    # Build composite: [template overlay | selfie | generated | aligned overlay]
    t = cv2.imread(str(tmpl_overlay))
    s = cv2.imread(str(selfie_path))
    g = cv2.imread(str(generated_path))
    a = cv2.imread(str(aligned_overlay))
    if any(x is None for x in [t, s, g, a]):
        raise RuntimeError('Failed to read one of composite inputs')

    # Resize to same height
    h = 420
    def rh(img):
        h0, w0 = img.shape[:2]
        return cv2.resize(img, (int(w0 * h / h0), h))
    imgs = [rh(x) for x in [t, s, g, a]]
    gaps = 10
    total_w = sum(im.shape[1] for im in imgs) + gaps * (len(imgs) + 1)
    canvas = np.ones((h + 50, total_w, 3), dtype=np.uint8) * 255
    x = gaps
    labels = ['template (overlays)', 'selfie', 'generated', 'aligned (overlays)']
    for im, lab in zip(imgs, labels):
        canvas[40:40 + h, x:x + im.shape[1]] = im
        cv2.putText(canvas, lab, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        x += im.shape[1] + gaps
    cv2.imwrite(str(out_path), canvas)


def main():
    template = Path('characters/Ali_F.png')
    selfie = Path('selfies_samples/005_Selfie_12.jpg')
    out_dir = Path('debug/consistency_run')
    # Clean dir
    if out_dir.exists():
        for p in out_dir.glob('*'):
            if p.is_dir():
                for q in p.rglob('*'):
                    q.unlink()
                p.rmdir()
            else:
                p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    build_composite(template, selfie, out_dir / 'composite_insightface.png', 'insightface')
    build_composite(template, selfie, out_dir / 'composite_mediapipe.png', 'mediapipe')


if __name__ == '__main__':
    main()



