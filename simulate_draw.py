#!/usr/bin/env python3
"""
Render robotic-arm vector paths (in mm) into a simulated drawing PNG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


PointMM = Tuple[float, float]
PathMM = List[PointMM]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render paths_mm.json into a simulated drawing image."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/paths_mm.json"),
        help="Input JSON from face_to_vectors.py (default: output/paths_mm.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/simulated_draw.png"),
        help="Output PNG path (default: output/simulated_draw.png).",
    )
    parser.add_argument(
        "--canvas-width",
        type=int,
        default=1200,
        help="Output image width in pixels (default: 1200).",
    )
    parser.add_argument(
        "--canvas-height",
        type=int,
        default=1200,
        help="Output image height in pixels (default: 1200).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=40,
        help="Margin around drawing area in pixels (default: 40).",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Stroke width in pixels (default: 2).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the rendered image in a window.",
    )
    return parser.parse_args()


def load_paths(input_path: Path) -> Tuple[float, float, List[PathMM]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    paper = data.get("paper_mm", {})
    paper_w = float(paper.get("width", 0.0))
    paper_h = float(paper.get("height", 0.0))
    if paper_w <= 0.0 or paper_h <= 0.0:
        raise ValueError("Invalid or missing paper_mm width/height in JSON.")

    raw_paths = data.get("paths", [])
    paths: List[PathMM] = []
    for raw_path in raw_paths:
        path: PathMM = []
        for point in raw_path:
            x = float(point["x_mm"])
            y = float(point["y_mm"])
            path.append((x, y))
        if len(path) >= 2:
            paths.append(path)
    return paper_w, paper_h, paths


def map_mm_to_px(
    point_mm: PointMM,
    paper_w_mm: float,
    paper_h_mm: float,
    canvas_w_px: int,
    canvas_h_px: int,
    margin_px: int,
) -> Tuple[int, int]:
    drawable_w = canvas_w_px - 2 * margin_px
    drawable_h = canvas_h_px - 2 * margin_px
    if drawable_w <= 0 or drawable_h <= 0:
        raise ValueError("Canvas is too small for the requested margin.")

    scale = min(drawable_w / paper_w_mm, drawable_h / paper_h_mm)
    x_offset = (canvas_w_px - paper_w_mm * scale) / 2.0
    y_offset = (canvas_h_px - paper_h_mm * scale) / 2.0

    x_mm, y_mm = point_mm
    x_px = int(round(x_offset + x_mm * scale))
    y_px = int(round(canvas_h_px - (y_offset + y_mm * scale)))
    return x_px, y_px


def draw_paths(
    image: np.ndarray,
    paths: Iterable[PathMM],
    paper_w_mm: float,
    paper_h_mm: float,
    margin_px: int,
    line_width: int,
) -> Tuple[int, float]:
    segment_count = 0
    length_mm = 0.0
    canvas_h, canvas_w = image.shape[:2]

    for path in paths:
        pts_px = [
            map_mm_to_px(
                pt,
                paper_w_mm=paper_w_mm,
                paper_h_mm=paper_h_mm,
                canvas_w_px=canvas_w,
                canvas_h_px=canvas_h,
                margin_px=margin_px,
            )
            for pt in path
        ]
        for i in range(1, len(path)):
            cv2.line(image, pts_px[i - 1], pts_px[i], color=(0, 0, 0), thickness=line_width)
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            length_mm += (dx * dx + dy * dy) ** 0.5
            segment_count += 1
    return segment_count, length_mm


def main() -> None:
    args = parse_args()

    if args.canvas_width <= 0 or args.canvas_height <= 0:
        raise ValueError("Canvas width/height must be positive.")
    if args.line_width <= 0:
        raise ValueError("Line width must be positive.")

    paper_w_mm, paper_h_mm, paths = load_paths(args.input)
    canvas = np.full((args.canvas_height, args.canvas_width, 3), 255, dtype=np.uint8)

    segment_count, length_mm = draw_paths(
        image=canvas,
        paths=paths,
        paper_w_mm=paper_w_mm,
        paper_h_mm=paper_h_mm,
        margin_px=args.margin,
        line_width=args.line_width,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), canvas)

    print(f"Rendered paths: {len(paths)}")
    print(f"Rendered segments: {segment_count}")
    print(f"Total pen-down length (mm): {length_mm:.2f}")
    print(f"Saved simulated drawing: {args.output}")

    if args.show:
        cv2.imshow("Simulated Draw", canvas)
        print("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
