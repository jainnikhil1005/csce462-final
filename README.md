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

## Main outputs

- `output/paths_mm.json`: vector paths in millimeter coordinates
- `output/preview.svg`: SVG preview of paths
- `output/focus_mask.png`: face-priority mask
- `output/foreground_mask.png`: foreground segmentation mask
- `output/region_mask.png`: combined mask used for edge extraction
- `output/keep_mask.png`: stricter mask used for contour filtering
- `output/line_art_binary.png`: threshold/edge map used for contour extraction
- `output/vector_overlay.png`: extracted contours overlay

## Simulate the drawing (before hardware)

Render the vector JSON into a PNG:

```bash
./.venv/bin/python simulate_draw.py --input output/paths_mm.json --output output/simulated_draw.png
```

Optional preview window:

```bash
./.venv/bin/python simulate_draw.py --input output/paths_mm.json --output output/simulated_draw.png --show
```

## Useful tuning flags

- `--paper-width-mm`, `--paper-height-mm`: target drawing area
- `--trace-mode`: `outline` (silhouette) or `detailed` (internal features)
- `--min-contour-area`, `--min-contour-length`: remove tiny/noisy paths
- `--epsilon-factor`: contour simplification amount
- `--border-margin`: remove contours touching image border (helps kill background lines)
- `--max-paths`: cap path count for faster drawing
- `--face-padding`: crop margin around detected face
- `simulate_draw.py --canvas-width --canvas-height --margin --line-width`: control simulation image size/look
