#!/usr/bin/env python3
"""
Capture (or load) a face image and convert it into drawable vector paths.
Outputs:
  - JSON paths in millimeters for robotic arm control
  - SVG preview of generated vectors
  - Optional debug images
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]
PathPoints = List[Point]


@dataclass
class PaperConfig:
    width_mm: float
    height_mm: float


@dataclass
class FaceCropResult:
    image_bgr: np.ndarray
    face_bbox: Optional[Tuple[int, int, int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a face photo and convert it into vector paths."
    )
    src_group = parser.add_mutually_exclusive_group(required=False)
    src_group.add_argument(
        "--input",
        type=Path,
        help="Path to an existing input image. If omitted, webcam capture is used.",
    )
    src_group.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index to use for capture (default: 0).",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("output"),
        help="Output directory for JSON/SVG/debug assets.",
    )
    parser.add_argument(
        "--paper-width-mm",
        type=float,
        default=200.0,
        help="Drawing area width in millimeters (default: 200).",
    )
    parser.add_argument(
        "--paper-height-mm",
        type=float,
        default=200.0,
        help="Drawing area height in millimeters (default: 200).",
    )
    parser.add_argument(
        "--face-padding",
        type=float,
        default=0.18,
        help="Padding fraction around detected face box (default: 0.18).",
    )
    parser.add_argument(
        "--min-contour-area",
        type=float,
        default=32.0,
        help="Minimum contour area in pixels to keep (default: 32).",
    )
    parser.add_argument(
        "--min-contour-length",
        type=float,
        default=10.0,
        help="Minimum contour perimeter in pixels to keep (default: 10.0).",
    )
    parser.add_argument(
        "--epsilon-factor",
        type=float,
        default=0.0035,
        help="Polyline simplification factor relative to contour perimeter (default: 0.0035).",
    )
    parser.add_argument(
        "--border-margin",
        type=int,
        default=6,
        help="Drop contours touching this many pixels from image border (default: 6).",
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=800,
        help="Max number of paths to keep after sorting (default: 800).",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Display generated edge map and vector overlay previews.",
    )
    parser.add_argument(
        "--trace-mode",
        choices=("outline", "detailed"),
        default="detailed",
        help="Tracing strategy: outline (face silhouette) or detailed (internal facial lines). Default: detailed.",
    )

    return parser.parse_args()


def capture_image(camera_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}.")

    print("Camera opened. Press SPACE to capture, or Q to quit.")
    captured = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        preview = frame.copy()
        cv2.putText(
            preview,
            "SPACE: capture | Q: quit",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Face Capture", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            captured = frame
            break
        if key in (ord("q"), 27):  # q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured is None:
        raise RuntimeError("Capture cancelled.")
    return captured


def detect_and_crop_face(image_bgr: np.ndarray, padding_frac: float) -> FaceCropResult:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        h, w = image_bgr.shape[:2]
        side = int(min(h, w) * 0.82)
        side = max(side, 200)
        cx, cy = w // 2, h // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)
        crop = image_bgr[y0:y1, x0:x1]
        print("No face detected. Using centered fallback crop.")
        return FaceCropResult(image_bgr=crop, face_bbox=None)

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_w = int(w * padding_frac)
    pad_h = int(h * padding_frac)

    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(image_bgr.shape[1], x + w + pad_w)
    y1 = min(image_bgr.shape[0], y + h + pad_h)
    crop = image_bgr[y0:y1, x0:x1]
    face_bbox_in_crop = (x - x0, y - y0, w, h)
    return FaceCropResult(image_bgr=crop, face_bbox=face_bbox_in_crop)


def build_focus_mask(
    image_shape: Tuple[int, int],
    face_bbox: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if face_bbox is not None:
        x, y, fw, fh = face_bbox
        cx = int(x + fw * 0.50)
        cy = int(y + fh * 0.56)
        ax = int(fw * 0.56)
        ay = int(fh * 0.84)
    else:
        cx = w // 2
        cy = int(h * 0.58)
        ax = int(w * 0.25)
        ay = int(h * 0.36)

    ax = max(20, min(ax, int(w * 0.38)))
    ay = max(20, min(ay, int(h * 0.43)))
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    _, hard_mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    hard_mask = cv2.erode(hard_mask, np.ones((9, 9), dtype=np.uint8), iterations=1)
    return hard_mask


def build_foreground_mask(
    face_bgr: np.ndarray,
    face_bbox: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    h, w = face_bgr.shape[:2]
    if face_bbox is None:
        return np.full((h, w), 255, dtype=np.uint8)

    x, y, fw, fh = face_bbox
    rx = max(1, int(x - fw * 0.18))
    ry = max(1, int(y - fh * 0.20))
    rw = min(w - rx - 1, int(fw * 1.36))
    rh = min(h - ry - 1, int(fh * 1.42))
    if rw < 10 or rh < 10:
        return np.full((h, w), 255, dtype=np.uint8)

    mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            face_bgr,
            mask,
            (rx, ry, rw, rh),
            bgd_model,
            fgd_model,
            3,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return np.full((h, w), 255, dtype=np.uint8)

    fg = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    kernel_small = np.ones((5, 5), dtype=np.uint8)
    kernel_big = np.ones((9, 9), dtype=np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_small, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_big, iterations=1)
    return fg


def roberts_edge(image: np.ndarray, thresh: Optional[float] = None) -> np.ndarray:
    """
    Apply the Roberts Cross edge detector as analogous to MATLAB's edge(..., 'roberts').
    """
    img_float = image.astype(np.float32)
    
    # Roberts Cross kernels
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    grad_x = cv2.filter2D(img_float, -1, kernel_x)
    grad_y = cv2.filter2D(img_float, -1, kernel_y)
    
    # Magnitude
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    if thresh is None:
        # Heuristic threshold based on the mean of non-zero gradients
        non_zero = magnitude[magnitude > 0]
        if len(non_zero) > 0:
            # Increased threshold multiplier to 2.5 to drop weak noisy edges (like skin texture)
            thresh = float(np.mean(non_zero)) * 2.5
        else:
            thresh = 40.0
            
    # Thresholding
    _, binary_edges = cv2.threshold(magnitude, thresh, 255, cv2.THRESH_BINARY)
    return binary_edges.astype(np.uint8)


def make_line_art_binary(face_bgr: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    # 1. Bilateral filter removes skin texture but preserves strong structural edges.
    blur_bgr = face_bgr.copy()
    for _ in range(3):
        blur_bgr = cv2.bilateralFilter(blur_bgr, d=11, sigmaColor=100, sigmaSpace=100)
    
    # 2. Convert to HSV for skin thresholding
    hsv = cv2.cvtColor(blur_bgr, cv2.COLOR_BGR2HSV)
    
    # Define bounds in OpenCV HSV format (H: 0-179, S: 0-255, V: 0-255)
    lower_hsv1 = np.array([0, 51, 0])
    upper_hsv1 = np.array([14, 255, 255])
    
    lower_hsv2 = np.array([165, 51, 0])
    upper_hsv2 = np.array([179, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
    mask2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
    skin_mask = cv2.bitwise_or(mask1, mask2)
    
    # 3. Combine with original region mask (focus + grabcut)
    final_mask = cv2.bitwise_and(skin_mask, region_mask)
    
    # 4. Extract Face Outline (Silhouette) using Roberts Cross
    gray_blur = cv2.cvtColor(blur_bgr, cv2.COLOR_BGR2GRAY)
    im_face = cv2.bitwise_and(gray_blur, gray_blur, mask=final_mask)
    outline_edges = roberts_edge(im_face, thresh=30.0)
    
    # 5. Extract Internal Features (Eyes, Nose, Lips) using a "Sketch" effect!
    # A block adaptive threshold works incredibly well for portrait drawing.
    gray_original = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray_light_blur = cv2.medianBlur(gray_original, 5)  # remove sensor noise only
    
    # Adaptive threshold creates a pen-sketch effect mimicking lighting shadows
    sketch = cv2.adaptiveThreshold(
        gray_light_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=6
    )
    
    # Clean up the sketch slightly to remove tiny disconnected specks
    sketch = cv2.morphologyEx(sketch, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    
    # Erode the bounding mask heavily so the sketch doesn't overlap the silhouette boundary
    eroded_mask = cv2.erode(final_mask, np.ones((13, 13), np.uint8), iterations=1)
    internal_features = cv2.bitwise_and(sketch, sketch, mask=eroded_mask)
    
    # 6. Combine Outline + Internal Features
    combined = cv2.bitwise_or(outline_edges, internal_features)
    
    # Optional thinning or cleanup can happen here, but we leave it thin for extraction
    return combined


def extract_outline_paths(
    region_mask: np.ndarray,
    epsilon_factor: float,
    border_margin: int,
) -> List[np.ndarray]:
    h, w = region_mask.shape[:2]
    cleaned = cv2.morphologyEx(
        region_mask,
        cv2.MORPH_CLOSE,
        np.ones((11, 11), dtype=np.uint8),
        iterations=2,
    )
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    center = np.array([w * 0.5, h * 0.58], dtype=np.float32)
    min_area = 0.06 * float(h * w)
    candidates: List[Tuple[float, np.ndarray]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        if (
            x <= border_margin
            or y <= border_margin
            or x + cw >= (w - border_margin)
            or y + ch >= (h - border_margin)
        ):
            continue

        moments = cv2.moments(contour)
        if moments["m00"] > 0.0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx, cy = contour[0, 0]
        dist = float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - center))
        score = area - 4.0 * dist
        candidates.append((score, contour))

    if candidates:
        best = max(candidates, key=lambda c: c[0])[1]
    else:
        best = max(contours, key=cv2.contourArea)

    peri = cv2.arcLength(best, closed=True)
    epsilon = max(0.5, epsilon_factor * peri * 0.4)
    approx = cv2.approxPolyDP(best, epsilon, closed=True)
    if len(approx) < 2:
        return []
    if not np.array_equal(approx[0, 0], approx[-1, 0]):
        approx = np.vstack([approx, approx[0:1]])
    return [approx]


def extract_paths(
    binary: np.ndarray,
    min_contour_area: float,
    min_contour_length: float,
    epsilon_factor: float,
    border_margin: int,
    max_paths: int,
    keep_mask: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    h, w = binary.shape[:2]

    kept: List[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue

        if keep_mask is not None:
            moments = cv2.moments(contour)
            if moments["m00"] > 0.0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx = int(contour[0, 0, 0])
                cy = int(contour[0, 0, 1])
            if cx < 0 or cy < 0 or cx >= keep_mask.shape[1] or cy >= keep_mask.shape[0]:
                continue
            if keep_mask[cy, cx] == 0:
                continue
            contour_pts = contour[:, 0, :]
            inside_ratio = np.mean(keep_mask[contour_pts[:, 1], contour_pts[:, 0]] > 0)
            if inside_ratio < 0.88:
                continue

        xs = contour[:, :, 0]
        ys = contour[:, :, 1]
        touches_border = (
            np.any(xs <= border_margin)
            or np.any(xs >= (w - 1 - border_margin))
            or np.any(ys <= border_margin)
            or np.any(ys >= (h - 1 - border_margin))
        )
        if touches_border:
            continue

        peri = cv2.arcLength(contour, closed=False)
        if peri < min_contour_length:
            continue
        epsilon = max(0.5, epsilon_factor * peri)
        approx = cv2.approxPolyDP(contour, epsilon, closed=False)
        if len(approx) >= 2:
            kept.append(approx)

    kept.sort(key=lambda c: cv2.arcLength(c, closed=False), reverse=True)
    return kept[:max_paths]


def map_paths_to_mm(
    contours: Sequence[np.ndarray],
    image_shape: Tuple[int, int],
    paper: PaperConfig,
) -> List[PathPoints]:
    h_px, w_px = image_shape
    scale = min(paper.width_mm / w_px, paper.height_mm / h_px)
    x_offset = (paper.width_mm - w_px * scale) / 2.0
    y_offset = (paper.height_mm - h_px * scale) / 2.0

    mapped: List[PathPoints] = []
    for contour in contours:
        path: PathPoints = []
        for pt in contour:
            x_px, y_px = pt[0]
            x_mm = x_px * scale + x_offset
            y_mm = (h_px - 1 - y_px) * scale + y_offset
            path.append((float(x_mm), float(y_mm)))
        if len(path) >= 2:
            mapped.append(path)
    return mapped


def save_svg(paths_mm: Sequence[PathPoints], paper: PaperConfig, out_path: Path) -> None:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{paper.width_mm}mm" height="{paper.height_mm}mm" viewBox="0 0 {paper.width_mm} {paper.height_mm}">',
        '<g fill="none" stroke="black" stroke-width="0.35" stroke-linecap="round" stroke-linejoin="round">',
    ]

    for path in paths_mm:
        points = " ".join(f"{x:.2f},{paper.height_mm - y:.2f}" for x, y in path)
        lines.append(f'<polyline points="{points}" />')

    lines.append("</g>")
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_json(
    paths_mm: Sequence[PathPoints],
    paper: PaperConfig,
    source_width_px: int,
    source_height_px: int,
    out_path: Path,
) -> None:
    payload = {
        "paper_mm": {"width": paper.width_mm, "height": paper.height_mm},
        "source_px": {"width": source_width_px, "height": source_height_px},
        "paths": [
            [{"x_mm": round(x, 3), "y_mm": round(y, 3)} for x, y in path]
            for path in paths_mm
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def draw_debug_overlay(binary: np.ndarray, contours: Sequence[np.ndarray]) -> np.ndarray:
    overlay = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, contours, contourIdx=-1, color=(0, 255, 0), thickness=1)
    return overlay


def main() -> None:
    args = parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.input is not None:
        image_bgr = cv2.imread(str(args.input))
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read input image: {args.input}")
    else:
        image_bgr = capture_image(args.camera_index)
        cv2.imwrite(str(outdir / "captured.jpg"), image_bgr)

    crop_result = detect_and_crop_face(image_bgr, args.face_padding)
    face_bgr = crop_result.image_bgr
    cv2.imwrite(str(outdir / "face_crop.jpg"), face_bgr)

    focus_mask = build_focus_mask(face_bgr.shape[:2], crop_result.face_bbox)
    cv2.imwrite(str(outdir / "focus_mask.png"), focus_mask)

    foreground_mask = build_foreground_mask(face_bgr, crop_result.face_bbox)
    cv2.imwrite(str(outdir / "foreground_mask.png"), foreground_mask)

    region_mask = cv2.bitwise_and(focus_mask, foreground_mask)
    cv2.imwrite(str(outdir / "region_mask.png"), region_mask)
    keep_mask = cv2.erode(focus_mask, np.ones((21, 21), dtype=np.uint8), iterations=1)
    cv2.imwrite(str(outdir / "keep_mask.png"), keep_mask)

    if args.trace_mode == "outline":
        binary = cv2.morphologyEx(
            region_mask,
            cv2.MORPH_GRADIENT,
            np.ones((5, 5), dtype=np.uint8),
        )
        contours = extract_outline_paths(
            region_mask=region_mask,
            epsilon_factor=args.epsilon_factor,
            border_margin=args.border_margin,
        )
    else:
        binary = make_line_art_binary(face_bgr, region_mask)
        contours = extract_paths(
            binary=binary,
            min_contour_area=args.min_contour_area,
            min_contour_length=args.min_contour_length,
            epsilon_factor=args.epsilon_factor,
            border_margin=args.border_margin,
            max_paths=args.max_paths,
            keep_mask=keep_mask,
        )

    cv2.imwrite(str(outdir / "line_art_binary.png"), binary)

    paper = PaperConfig(width_mm=args.paper_width_mm, height_mm=args.paper_height_mm)
    paths_mm = map_paths_to_mm(contours, face_bgr.shape[:2], paper)

    json_path = outdir / "paths_mm.json"
    svg_path = outdir / "preview.svg"
    save_json(paths_mm, paper, face_bgr.shape[1], face_bgr.shape[0], json_path)
    save_svg(paths_mm, paper, svg_path)

    debug_overlay = draw_debug_overlay(binary, contours)
    cv2.imwrite(str(outdir / "vector_overlay.png"), debug_overlay)

    print(f"Generated {len(paths_mm)} vector paths.")
    print(f"Trace mode:       {args.trace_mode}")
    print(f"JSON path output: {json_path}")
    print(f"SVG preview:      {svg_path}")
    print(f"Focus mask:       {outdir / 'focus_mask.png'}")
    print(f"Foreground mask:  {outdir / 'foreground_mask.png'}")
    print(f"Region mask:      {outdir / 'region_mask.png'}")
    print(f"Keep mask:        {outdir / 'keep_mask.png'}")
    print(f"Debug binary:     {outdir / 'line_art_binary.png'}")
    print(f"Debug overlay:    {outdir / 'vector_overlay.png'}")

    if args.show_preview:
        cv2.imshow("Line Art Binary", binary)
        cv2.imshow("Vector Overlay", debug_overlay)
        print("Press any key on an image window to close previews.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
