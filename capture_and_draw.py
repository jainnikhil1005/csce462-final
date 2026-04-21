#!/usr/bin/env python3
"""
capture_and_draw.py — Full pipeline: capture face → generate vector paths → draw with motors.
Run this on the Raspberry Pi.

Steps:
  1. Capture face photo from webcam
  2. Process into vector paths (runs face_to_vectors.py as a subprocess)
  3. Confirm with the user before drawing
  4. Drive the stepper motors to draw the face
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep
from typing import List, Tuple

import RPi.GPIO as GPIO

# ============================================================
#  MOTOR CONFIGURATION — adjust to match your hardware
# ============================================================
STEPS_PER_MM       = 80.0   # steps per mm of travel — calibrate this!
DRAW_SPEED_MM_S    = 15.0   # pen-down drawing speed  (mm/s)
TRAVEL_SPEED_MM_S  = 40.0   # pen-up travel speed     (mm/s)

# X-axis motor (motor 0)
X_PUL, X_DIR, X_ENA = 19, 13, 17
X_INVERT = False             # set True if X moves the wrong direction

# Y-axis motor (motor 1)
Y_PUL, Y_DIR, Y_ENA = 12, 20, 21
Y_INVERT = False             # set True if Y moves the wrong direction

# Pen toggle — GPIO 16, one 1-second HIGH pulse toggles pen up/down
PEN_PIN      = 16
PEN_PULSE_S  = 1.0
PEN_SETTLE_S = 0.3

# ============================================================
#  PIPELINE CONFIGURATION
# ============================================================
OUTDIR       = Path("output")
PATHS_JSON   = OUTDIR / "paths_mm.json"
TRACE_MODE   = "cartoon"     # cartoon | detailed | outline | landmarks
PAPER_MM     = 200.0         # drawing area size in mm (square)
# ============================================================

PointMM = Tuple[float, float]


# ------------------------------------------------------------------
# STEP 1 — Capture face and generate vector paths
# ------------------------------------------------------------------

def generate_paths() -> List[List[PointMM]]:
    """Run face_to_vectors.py to capture the face and write paths_mm.json."""
    print("\n[1/3] Capturing face and generating vector paths...")

    cmd = [
        sys.executable, "face_to_vectors.py",
        "--outdir",          str(OUTDIR),
        "--trace-mode",      TRACE_MODE,
        "--paper-width-mm",  str(PAPER_MM),
        "--paper-height-mm", str(PAPER_MM),
        "--pi-camera",
    ]

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print("[!] face_to_vectors.py failed.")
        sys.exit(1)

    paths = load_paths(PATHS_JSON)
    print(f"    Generated {len(paths)} paths  →  {PATHS_JSON}")
    return paths


def load_paths(json_path: Path) -> List[List[PointMM]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    paths: List[List[PointMM]] = []
    for raw_path in data.get("paths", []):
        path = [(float(pt["x_mm"]), float(pt["y_mm"])) for pt in raw_path]
        if len(path) >= 2:
            paths.append(path)
    return paths


# ------------------------------------------------------------------
# STEP 2 — Confirm before drawing
# ------------------------------------------------------------------

def confirm(paths: List[List[PointMM]]) -> None:
    """Show stats and wait for the user to confirm before starting motors."""
    total_pts = sum(len(p) for p in paths)
    total_len = sum(
        math.hypot(p[i][0] - p[i-1][0], p[i][1] - p[i-1][1])
        for p in paths for i in range(1, len(p))
    )
    est_draw_s  = total_len / DRAW_SPEED_MM_S
    est_total_s = est_draw_s * 1.25  # rough estimate including travel

    print(f"\n[2/3] Ready to draw:")
    print(f"      Paths:          {len(paths)}")
    print(f"      Total points:   {total_pts}")
    print(f"      Pen-down length:{total_len:.0f} mm")
    print(f"      Est. draw time: {est_total_s / 60:.1f} min")
    print(f"\n      Check output/vector_overlay.png to preview the drawing.")
    input("\n      Press Enter to start drawing  (pen UP, carriage at origin)...")


# ------------------------------------------------------------------
# STEP 3 — Motor control
# ------------------------------------------------------------------

def gpio_setup() -> None:
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in (X_PUL, X_DIR, X_ENA, Y_PUL, Y_DIR, Y_ENA, PEN_PIN):
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)


def motors_enable(on: bool) -> None:
    GPIO.output(X_ENA, GPIO.HIGH if on else GPIO.LOW)
    GPIO.output(Y_ENA, GPIO.HIGH if on else GPIO.LOW)


def pen_toggle() -> None:
    GPIO.output(PEN_PIN, GPIO.HIGH)
    sleep(PEN_PULSE_S)
    GPIO.output(PEN_PIN, GPIO.LOW)
    sleep(PEN_SETTLE_S)


def move_steps(dx: int, dy: int, speed_mm_s: float) -> None:
    """Move dx, dy steps simultaneously using Bresenham's algorithm."""
    if dx == 0 and dy == 0:
        return

    GPIO.output(X_DIR, GPIO.LOW if (dx >= 0) ^ X_INVERT else GPIO.HIGH)
    GPIO.output(Y_DIR, GPIO.LOW if (dy >= 0) ^ Y_INVERT else GPIO.HIGH)

    abs_x = abs(dx)
    abs_y = abs(dy)
    n_ticks = max(abs_x, abs_y)

    dist_mm = math.hypot(dx, dy) / STEPS_PER_MM
    half_period = dist_mm / speed_mm_s / n_ticks / 2.0

    err_x = n_ticks // 2
    err_y = n_ticks // 2
    t = perf_counter()

    for _ in range(n_ticks):
        px = False
        py = False

        err_x -= abs_x
        if err_x < 0:
            err_x += n_ticks
            px = True

        err_y -= abs_y
        if err_y < 0:
            err_y += n_ticks
            py = True

        if px:
            GPIO.output(X_PUL, GPIO.HIGH)
        if py:
            GPIO.output(Y_PUL, GPIO.HIGH)

        t += half_period
        while perf_counter() < t:
            pass

        if px:
            GPIO.output(X_PUL, GPIO.LOW)
        if py:
            GPIO.output(Y_PUL, GPIO.LOW)

        t += half_period
        while perf_counter() < t:
            pass


def move_to(pos: PointMM, target: PointMM, speed_mm_s: float) -> PointMM:
    dx_steps = round((target[0] - pos[0]) * STEPS_PER_MM)
    dy_steps = round((target[1] - pos[1]) * STEPS_PER_MM)
    move_steps(dx_steps, dy_steps, speed_mm_s)
    return (
        pos[0] + dx_steps / STEPS_PER_MM,
        pos[1] + dy_steps / STEPS_PER_MM,
    )


def draw(paths: List[List[PointMM]]) -> None:
    print(f"\n[3/3] Drawing {len(paths)} paths...")
    gpio_setup()
    motors_enable(True)

    pen_is_down = False
    pos: PointMM = (0.0, 0.0)

    try:
        for i, path in enumerate(paths):
            print(f"  Path {i + 1}/{len(paths)}  ({len(path)} pts)", end="\r")

            # Pen up, travel to start of path
            if pen_is_down:
                pen_toggle()
                pen_is_down = False
            pos = move_to(pos, path[0], TRAVEL_SPEED_MM_S)

            # Pen down, draw each segment
            pen_toggle()
            pen_is_down = True
            for pt in path[1:]:
                pos = move_to(pos, pt, DRAW_SPEED_MM_S)

        # Finished — lift pen and return home
        if pen_is_down:
            pen_toggle()

        print(f"\n  Returning to origin from ({pos[0]:.1f}, {pos[1]:.1f}) mm ...")
        pos = move_to(pos, (0.0, 0.0), TRAVEL_SPEED_MM_S)
        print("  Done!")

    except KeyboardInterrupt:
        print("\n[!] Interrupted — lifting pen.")
        if pen_is_down:
            pen_toggle()

    finally:
        motors_enable(False)
        GPIO.cleanup()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print("=== Face Drawing Robot ===")

    # Allow skipping capture if paths_mm.json already exists
    if PATHS_JSON.exists():
        choice = input(f"\nFound existing {PATHS_JSON}. Use it? [y/N]: ").strip().lower()
        if choice == "y":
            paths = load_paths(PATHS_JSON)
            print(f"    Loaded {len(paths)} paths from existing file.")
        else:
            paths = generate_paths()
    else:
        paths = generate_paths()

    if not paths:
        print("[!] No paths to draw.")
        sys.exit(1)

    confirm(paths)
    draw(paths)


if __name__ == "__main__":
    main()
