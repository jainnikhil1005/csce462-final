# Face To Vectors (Robotic Arm Prep)

This script captures your face from webcam (or uses an input image) and converts it into drawable vector paths.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
./.venv/bin/python -m pip install -r requirements.txt
```

## Run

Webcam capture:

```bash
./.venv/bin/python face_to_vectors.py --outdir output --show-preview
```

Use an existing image:

```bash
./.venv/bin/python face_to_vectors.py --input my_face.jpg --outdir output
```

Outline-only tracing (recommended for robotic portrait silhouette):

```bash
./.venv/bin/python face_to_vectors.py --outdir output --trace-mode outline
```

Detailed tracing (eyes/nose/lips + more interior lines):

```bash
./.venv/bin/python face_to_vectors.py --outdir output --trace-mode detailed
```

`detailed` now biases extraction toward eyes, brows, nose, and mouth so the portrait keeps more of the structure that affects likeness.

Cartoon tracing based on the DinjanAI/Image_to_Cartoon OpenCV pipeline:

```bash
./.venv/bin/python face_to_vectors.py --outdir output --trace-mode cartoon
```

This mode first cartoonizes the webcam photo with adaptive-threshold edges plus bilateral smoothing, then extracts vectors from that cleaned portrait.

Landmark tracing with MediaPipe Face Mesh (best likeness when the dependency is installed):

```bash
./.venv/bin/python face_to_vectors.py --outdir output --trace-mode landmarks
```

If you run `landmarks` on macOS and see an OpenGL initialization error, run it from a normal desktop terminal session instead of a headless shell.

## Main outputs

- `output/paths_mm.json`: vector paths in millimeter coordinates
- `output/preview.svg`: SVG preview of paths
- `output/focus_mask.png`: face-priority mask
- `output/foreground_mask.png`: foreground segmentation mask
- `output/region_mask.png`: combined mask used for edge extraction
- `output/feature_mask.png`: eyes/nose/mouth priority mask used to preserve facial structure
- `output/keep_mask.png`: stricter mask used for contour filtering
- `output/cartoon_face.png`: cartoonized face image used by `--trace-mode cartoon`
- `output/line_art_binary.png`: threshold/edge map used for contour extraction
- `output/vector_overlay.png`: extracted contours overlay

## Simulate the drawing (before hardware)

Render the vector JSON into a PNG:

```bash
./.venv/bin/python simulate_draw.py --input output/paths_mm.json --output output/simulated_draw.png
```

The renderer now applies a face-focused cleanup pass by default so tiny stray vectors and weak edge clutter are dropped before previewing.

Optional preview window:

```bash
./.venv/bin/python simulate_draw.py --input output/paths_mm.json --output output/simulated_draw.png --show
```

Save the cleaned vector set for reuse:

```bash
./.venv/bin/python simulate_draw.py \
  --input output/paths_mm.json \
  --output output/simulated_draw.png \
  --filtered-json-output output/paths_filtered.json
```

## Useful tuning flags

- `--paper-width-mm`, `--paper-height-mm`: target drawing area
- `--trace-mode`: `outline` (silhouette), `detailed` (internal features), or `cartoon` (DinjanAI-style cartoon preprocessing)
- `--trace-mode landmarks`: use MediaPipe facial landmarks for more face-accurate contours
- `--min-contour-area`, `--min-contour-length`: remove tiny/noisy paths
- `--epsilon-factor`: contour simplification amount
- `--border-margin`: remove contours touching image border (helps kill background lines)
- `--max-paths`: cap path count for faster drawing
- `--face-padding`: crop margin around detected face
- `simulate_draw.py --canvas-width --canvas-height --margin --line-width`: control simulation image size/look
- `simulate_draw.py --min-path-length-mm --min-path-diagonal-mm --keep-length-ratio`: control how aggressively low-value vector paths are removed
- `simulate_draw.py --filtered-json-output`: write the cleaned vector paths back to JSON
