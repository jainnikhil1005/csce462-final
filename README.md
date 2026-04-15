# Face To Vectors (Robotic Arm Prep)

This script captures your face from webcam (or uses an input image) and converts it into drawable vector paths.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
./.venv/bin/python -m pip install -r requirements.txt
```

## Run

Recommended webcam command:

```bash
python face_to_vectors.py \
  --outdir output \
  --trace-mode cartoon \
  --show-preview
```

This is the default workflow for the project right now: capture from webcam, apply the DinjanAI/Image_to_Cartoon preprocessing, and preview the extracted vectors immediately.

Use an existing image instead of the webcam:

```bash
python face_to_vectors.py --input my_face.jpg --outdir output --trace-mode cartoon
```

Other trace modes:

```bash
python face_to_vectors.py --outdir output --trace-mode outline
```

- `detailed`: more internal facial lines, biased toward eyes, brows, nose, and mouth
- `landmarks`: MediaPipe Face Mesh contours, when that dependency initializes correctly on your machine
- `outline`: mostly silhouette / outer face shape

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

The renderer now defaults to the original source image size, zero margin, 1-pixel strokes, and no extra filtering so it matches `output/vector_overlay.png` much more closely.

Optional preview window:

```bash
./.venv/bin/python simulate_draw.py --input output/paths_mm.json --output output/simulated_draw.png --show
```

Render with a black background and green contour lines like the vector overlay:

```bash
./.venv/bin/python simulate_draw.py \
  --input output/paths_mm.json \
  --output output/simulated_draw.png \
  --overlay-style
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
- `simulate_draw.py --filter --min-path-length-mm --min-path-diagonal-mm --keep-length-ratio`: optionally remove low-value vector paths before rendering
- `simulate_draw.py --overlay-style`: use a black background and green lines like `vector_overlay.png`
- `simulate_draw.py --filtered-json-output`: write the cleaned vector paths back to JSON
