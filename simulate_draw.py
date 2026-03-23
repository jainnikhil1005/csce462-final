#!/usr/bin/env python3
"""
Render robotic-arm vector paths (in mm) into a simulated drawing PNG.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np


PointMM = Tuple[float, float]
PathMM = List[PointMM]


@dataclass
class PathStats:
    index: int
    path: PathMM
    length_mm: float
    bbox_diag_mm: float
    face_coverage: float
    border_distance_mm: float
    importance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render paths_mm.json into a simulated drawing image with optional face-focused filtering."
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
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable the built-in path filter and render every vector path.",
    )
    parser.add_argument(
        "--min-path-length-mm",
        type=float,
        default=14.0,
        help="Drop very short paths below this length when they also have tiny spread (default: 14.0).",
    )
    parser.add_argument(
        "--min-path-diagonal-mm",
        type=float,
        default=5.0,
        help="Drop paths with a tiny bounding-box diagonal unless they are long enough (default: 5.0).",
    )
    parser.add_argument(
        "--keep-length-ratio",
        type=float,
        default=0.88,
        help="After scoring paths, keep enough of them to preserve this fraction of total pen-down length (default: 0.88).",
    )
    parser.add_argument(
        "--min-keep-paths",
        type=int,
        default=7,
        help="Always keep at least this many of the best paths after filtering (default: 7).",
    )
    parser.add_argument(
        "--edge-margin-ratio",
        type=float,
        default=0.07,
        help="Treat paths near the drawing bounds as likely noise when they sit outside the face region (default: 0.07).",
    )
    parser.add_argument(
        "--filtered-json-output",
        type=Path,
        help="Optional path to save the filtered vector JSON for reuse outside the preview render.",
    )
    return parser.parse_args()


def load_paths(input_path: Path) -> Tuple[float, float, Optional[Tuple[int, int]], List[PathMM]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    paper = data.get("paper_mm", {})
    paper_w = float(paper.get("width", 0.0))
    paper_h = float(paper.get("height", 0.0))
    if paper_w <= 0.0 or paper_h <= 0.0:
        raise ValueError("Invalid or missing paper_mm width/height in JSON.")

    source = data.get("source_px", {})
    source_w = int(source.get("width", 0) or 0)
    source_h = int(source.get("height", 0) or 0)
    source_px = (source_w, source_h) if source_w > 0 and source_h > 0 else None

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
    return paper_w, paper_h, source_px, paths


def save_paths_json(
    output_path: Path,
    paper_w_mm: float,
    paper_h_mm: float,
    paths: List[PathMM],
    source_px: Optional[Tuple[int, int]],
) -> None:
    payload = {
        "paper_mm": {"width": paper_w_mm, "height": paper_h_mm},
        "paths": [
            [{"x_mm": round(x, 3), "y_mm": round(y, 3)} for x, y in path]
            for path in paths
        ],
    }
    if source_px is not None:
        payload["source_px"] = {"width": source_px[0], "height": source_px[1]}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def path_length_mm(path: PathMM) -> float:
    return sum(
        math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        for i in range(1, len(path))
    )


def path_bounds(path: PathMM) -> Tuple[float, float, float, float]:
    xs = [pt[0] for pt in path]
    ys = [pt[1] for pt in path]
    return min(xs), min(ys), max(xs), max(ys)


def all_paths_bounds(paths: List[PathMM]) -> Tuple[float, float, float, float]:
    mins_maxes = [path_bounds(path) for path in paths]
    return (
        min(bounds[0] for bounds in mins_maxes),
        min(bounds[1] for bounds in mins_maxes),
        max(bounds[2] for bounds in mins_maxes),
        max(bounds[3] for bounds in mins_maxes),
    )


def ellipse_distance(
    point: PointMM,
    center: PointMM,
    axes: PointMM,
) -> float:
    ax = max(axes[0], 1e-6)
    ay = max(axes[1], 1e-6)
    dx = (point[0] - center[0]) / ax
    dy = (point[1] - center[1]) / ay
    return math.hypot(dx, dy)


def filter_paths(
    paths: List[PathMM],
    min_path_length_mm: float,
    min_path_diagonal_mm: float,
    keep_length_ratio: float,
    min_keep_paths: int,
    edge_margin_ratio: float,
) -> Tuple[List[PathMM], int]:
    if len(paths) <= 1:
        return paths, 0

    x0, y0, x1, y1 = all_paths_bounds(paths)
    span_w = max(x1 - x0, 1.0)
    span_h = max(y1 - y0, 1.0)
    face_center = (x0 + span_w * 0.50, y0 + span_h * 0.56)
    face_axes = (span_w * 0.43, span_h * 0.52)
    edge_margin_mm = min(span_w, span_h) * max(edge_margin_ratio, 0.0)
    edge_keep_length_mm = max(min_path_length_mm * 1.8, 0.18 * (span_w + span_h))

    candidates: List[PathStats] = []
    hard_dropped = 0
    for index, path in enumerate(paths):
        min_x, min_y, max_x, max_y = path_bounds(path)
        bbox_diag_mm = math.hypot(max_x - min_x, max_y - min_y)
        length_mm = path_length_mm(path)
        face_coverage = float(
            np.mean(
                [
                    ellipse_distance(point, face_center, face_axes) <= 1.0
                    for point in path
                ]
            )
        )
        centroid = (
            sum(point[0] for point in path) / len(path),
            sum(point[1] for point in path) / len(path),
        )
        center_distance = ellipse_distance(centroid, face_center, face_axes)
        centrality = max(0.0, 1.0 - min(center_distance, 1.6) / 1.6)
        border_distance_mm = min(min_x - x0, x1 - max_x, min_y - y0, y1 - max_y)

        is_tiny = length_mm < min_path_length_mm and bbox_diag_mm < min_path_diagonal_mm
        is_edge_noise = (
            border_distance_mm < edge_margin_mm
            and face_coverage < 0.35
            and length_mm < edge_keep_length_mm
        )
        if is_tiny or is_edge_noise:
            hard_dropped += 1
            continue

        importance = (
            length_mm * (0.78 + 0.52 * face_coverage)
            + bbox_diag_mm * (0.40 + 0.80 * centrality)
        )
        candidates.append(
            PathStats(
                index=index,
                path=path,
                length_mm=length_mm,
                bbox_diag_mm=bbox_diag_mm,
                face_coverage=face_coverage,
                border_distance_mm=border_distance_mm,
                importance=importance,
            )
        )

    if not candidates:
        fallback = sorted(
            enumerate(paths),
            key=lambda item: path_length_mm(item[1]),
            reverse=True,
        )
        keep_count = min(len(fallback), max(1, min_keep_paths))
        kept = [path for _, path in sorted(fallback[:keep_count], key=lambda item: item[0])]
        return kept, len(paths) - len(kept)

    ranked = sorted(candidates, key=lambda stats: stats.importance, reverse=True)
    total_length_mm = sum(stats.length_mm for stats in ranked)
    target_length_mm = total_length_mm * min(max(keep_length_ratio, 0.0), 1.0)
    top_importance = ranked[0].importance

    kept_indexes: Set[int] = set()
    kept_length_mm = 0.0
    for position, stats in enumerate(ranked):
        must_keep = position < max(min_keep_paths, 1)
        high_value = stats.importance >= top_importance * 0.52
        if must_keep or kept_length_mm < target_length_mm or high_value:
            kept_indexes.add(stats.index)
            kept_length_mm += stats.length_mm
        else:
            break

    kept_paths = [path for index, path in enumerate(paths) if index in kept_indexes]
    return kept_paths, len(paths) - len(kept_paths)


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
            length_mm += math.hypot(dx, dy)
            segment_count += 1
    return segment_count, length_mm


def main() -> None:
    args = parse_args()

    if args.canvas_width <= 0 or args.canvas_height <= 0:
        raise ValueError("Canvas width/height must be positive.")
    if args.line_width <= 0:
        raise ValueError("Line width must be positive.")
    if args.min_path_length_mm < 0.0 or args.min_path_diagonal_mm < 0.0:
        raise ValueError("Path filtering thresholds must be non-negative.")
    if args.min_keep_paths < 1:
        raise ValueError("min-keep-paths must be at least 1.")

    paper_w_mm, paper_h_mm, source_px, paths = load_paths(args.input)
    original_path_count = len(paths)
    dropped_paths = 0
    if not args.no_filter:
        paths, dropped_paths = filter_paths(
            paths=paths,
            min_path_length_mm=args.min_path_length_mm,
            min_path_diagonal_mm=args.min_path_diagonal_mm,
            keep_length_ratio=args.keep_length_ratio,
            min_keep_paths=args.min_keep_paths,
            edge_margin_ratio=args.edge_margin_ratio,
        )

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
    if args.filtered_json_output is not None:
        args.filtered_json_output.parent.mkdir(parents=True, exist_ok=True)
        save_paths_json(args.filtered_json_output, paper_w_mm, paper_h_mm, paths, source_px)

    print(f"Input paths: {original_path_count}")
    if args.no_filter:
        print("Path filter: disabled")
    else:
        print(f"Filtered paths kept: {len(paths)}")
        print(f"Filtered paths dropped: {dropped_paths}")
    print(f"Rendered paths: {len(paths)}")
    print(f"Rendered segments: {segment_count}")
    print(f"Total pen-down length (mm): {length_mm:.2f}")
    print(f"Saved simulated drawing: {args.output}")
    if args.filtered_json_output is not None:
        print(f"Saved filtered JSON: {args.filtered_json_output}")

    if args.show:
        cv2.imshow("Simulated Draw", canvas)
        print("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
