## Sept 5 Refactor Plan: Clean, Readable Alignment Pipeline

### Goals
- Single, readable flow that matches intent: create template-sized canvas; rotate/scale/translate generated so eyes and mouth match template.
- Unified landmark detection with consistent detector selection (InsightFace or Mediapipe) across both images.
- Three-point similarity transform (L eye, R eye, mouth) for stable alignment.
- No overlays in production; optional debug overlay separate.
- Add bypass flags to test both detectors explicitly on the same pair and measure consistency.

### Steps
1) Implement detectors module with a unified interface and bypass flags
2) Implement transform module (compute 3-point similarity; warp)
3) Implement pipeline module with `select_consistent_landmarks(...)` and `align_generated_to_template(...)`
4) Create a test script that:
   - Processes one template/selfie pair twice (force InsightFace; force Mediapipe)
   - Saves aligned outputs
   - Re-detects landmarks on both outputs and the template using both detectors
   - Measures per-axis pixel deltas; assert |dx|,|dy| ≤ 5px for eye centers and mouth centers
5) Append results (logs and metrics) to this plan
6) Wire the server/batch to the new pipeline 

---

## Step 1: Detectors module - Implementation Result
Implemented: `alignment/detectors.py`

APIs (signatures):
```python
# alignment/detectors.py
@dataclass
class DetectedLandmarks:
    left_eye_center: np.ndarray
    right_eye_center: np.ndarray
    mouth_center: np.ndarray
    points5: np.ndarray
    method: str

class InsightFaceLandmarkDetector:
    def __init__(self, det_size: tuple[int,int] = (640, 640), det_thresh: float = 0.35): ...
    def detect(self, image_path: Path) -> DetectedLandmarks | None: ...

class MediapipeLandmarkDetector:
    def __init__(self): ...
    def detect(self, image_path: Path) -> DetectedLandmarks | None: ...

class CombinedLandmarkDetector:
    def __init__(self): ...
    def detect_both_images(
        self,
        template_path: Path,
        generated_path: Path,
        bypass_insightface: bool = False,
        bypass_mediapipe: bool = False,
    ) -> tuple[DetectedLandmarks | None, DetectedLandmarks | None, str]: ...
```

## Step 2: Transform module - Implementation Result
Implemented: `alignment/transform.py`

APIs (signatures):
```python
# alignment/transform.py
def compute_similarity_transform_from_three_points(
    template_points: np.ndarray,  # (3,2) left_eye, right_eye, mouth
    generated_points: np.ndarray, # (3,2)
) -> np.ndarray:                  # (2,3) affine
    ...

def warp_generated_to_template_canvas(
    generated_path: Path,
    template_size: tuple[int, int],  # (width, height)
    affine_2x3: np.ndarray,          # (2,3)
    output_path: Path,
) -> bool: ...
```

## Step 3: Pipeline module - Implementation Result
Planned file: `alignment/pipeline.py`

APIs (signatures):
```python
# alignment/pipeline.py
from alignment.detectors import CombinedLandmarkDetector, DetectedLandmarks
from alignment.transform import compute_similarity_transform_from_three_points, warp_generated_to_template_canvas

def select_consistent_landmarks(
    template_path: Path,
    generated_path: Path,
    disable_insightface: bool = False,
    disable_mediapipe: bool = False,
) -> tuple[DetectedLandmarks, DetectedLandmarks, str]:
    """Return landmarks for template and generated, using the same detector ('insightface' or 'mediapipe'). Raises on failure."""

def align_generated_to_template(
    generated_path: Path,
    template_path: Path,
    output_path: Path,
    disable_insightface: bool = False,
    disable_mediapipe: bool = False,
) -> dict:
    """Align generated to template using three-point similarity. Returns dict with keys: success, method, affine (2x3), output_path."""
```

Additional APIs (data and helpers):
```python
@dataclass
class AlignmentOutcome:
    success: bool
    method: str                # 'insightface' | 'mediapipe'
    affine_2x3: np.ndarray     # (2,3)
    template_size: tuple[int,int]
    output_path: Path

def render_landmarks_overlay(image_path: Path, landmarks: DetectedLandmarks, output_path: Path) -> None: ...
```

## Step 4: Single-pair Detector Consistency Test - Result
Planned files: `debug/test_alignment_consistency.py`

APIs (signatures):
```python
# debug/test_alignment_consistency.py
def run_single_pair_consistency_test(
    template_path: Path,
    selfie_path: Path,
    work_dir: Path,
) -> dict:
    """
    1) Produce generated image via StyleTransferRunner.run_style_transfer (or reuse an existing result)
    2) Align twice to template: once forcing InsightFace (disable_mediapipe=True), once forcing Mediapipe (disable_insightface=True)
    3) For both aligned outputs and the template, detect landmarks with BOTH detectors (independently)
    4) Measure per-axis deltas between Mediapipe vs InsightFace landmarks on template and on each aligned output
    5) Assert |dx|, |dy| <= 5 for eyes and mouth; record metrics and pass/fail
    Returns a dict with measurements and boolean pass flags
    """

def measure_per_axis_deltas(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (|dx|, |dy|) in pixels between two 2D points."""
```

Pass criteria:
- For template image and for each aligned output, mediapipe vs insightface deltas for left eye center, right eye center, and mouth center must each satisfy |dx| ≤ 5 and |dy| ≤ 5.

## Step 5: Summary and Next
Run log excerpt (Ali_F + 005_Selfie_12):

```json
{
  "align_insightface": { "success": true, "method": "insightface", "output_path": "debug/consistency_run/aligned_insightface.png" },
  "align_mediapipe":   { "success": true, "method": "mediapipe",   "output_path": "debug/consistency_run/aligned_mediapipe.png" },
  "metrics": {
    "template_mediapipe_vs_insightface": {
      "left_eye":  [10.7864, 1.2773],
      "right_eye": [4.8756, 0.5881],
      "mouth":     [5.3516, 3.9320]
    }
  },
  "pass_template": false
}
```

Observations:
- Template deltas fail the ≤5px threshold on left eye (dx≈10.79). This indicates detector disagreement on the stylized template (insightface vs mediapipe) and undermines cross-detector consistency, not the transform itself.

Next actions:
- Prefer a single detector per run (do not mix for evaluation). For alignment, use the detector that finds both faces; for evaluation, measure intra-detector self-consistency (pre vs post align) rather than cross-detector.
- Optionally tune Mediapipe indices or disable Mediapipe for highly stylized templates when InsightFace is stable.

---

## Current Code Inventory (for traceability)
- server.py → calls `ImageProcessor.process_images(...)`
- image_processor.py → generation (OpenRouter) + current alignment entry calling `face_alignment_unified.align_generated_image`
- face_alignment_unified.py → mixed detection (IF/MP) + 2-point style math (to be replaced)
- generate_all_results.py → batch driver; writes `results/` and table
- process_with_landmarks.py → debug visualization mixed with detection (to be moved to debug-only overlays)
- apply_alignment_to_existing.py → applies alignment to files in `results/` (to be adapted to new pipeline)
## Rename/Move Plan
- `image_processor.py` → `style_transfer_runner.py`
  - `request_style_transfer_image(template_path: Path, selfie_path: Path) -> dict`
  - `run_style_transfer_and_align(template_path: Path, selfie_path: Path, output_path: Path, disable_insightface=False, disable_mediapipe=False, debug_overlay_dir: Path | None = None) -> dict`
- `face_alignment_unified.py` → superseded by `alignment/{detectors,transform,pipeline}.py`
- Debug overlays moved to `debug/overlay.py` with `render_landmarks_overlay(...)` only; never write into `results/` in production.

