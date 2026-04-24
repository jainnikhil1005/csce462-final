#!/usr/bin/env python3
"""
draw_paths.py — Drive a 2-axis stepper plotter from paths_mm.json.
Run this on the Raspberry Pi.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from time import perf_counter, sleep
from typing import List, Tuple

import RPi.GPIO as GPIO

# ============================================================
#  CONFIGURATION — adjust these to match your hardware
# ============================================================
PATHS_JSON         = "output/paths_mm.json"

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
PEN_PULSE_S  = 0.05           # pulse duration (seconds)
PEN_SETTLE_S = 0.3           # pause after toggling before moving
# ============================================================

PointMM = Tuple[float, float]


def setup() -> None:
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in (X_PUL, X_DIR, X_ENA, Y_PUL, Y_DIR, Y_ENA, PEN_PIN):
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)


def motors_enable(on: bool) -> None:
    GPIO.output(X_ENA, GPIO.HIGH if on else GPIO.LOW)
    GPIO.output(Y_ENA, GPIO.HIGH if on else GPIO.LOW)


def pen_up() -> None:
    """Energise solenoid to lift pen. Stays HIGH until pen_down() is called."""
    GPIO.output(PEN_PIN, GPIO.HIGH)
    sleep(PEN_SETTLE_S)


def pen_down() -> None:
    """De-energise solenoid — spring drops pen onto paper."""
    GPIO.output(PEN_PIN, GPIO.LOW)
    sleep(PEN_SETTLE_S)


def move_steps(dx: int, dy: int, speed_mm_s: float) -> None:
    """
    Move dx steps on X and dy steps on Y simultaneously.
    Uses Bresenham's algorithm to interleave pulses on both axes.
    """
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
        pulse_x = False
        pulse_y = False

        err_x -= abs_x
        if err_x < 0:
            err_x += n_ticks
            pulse_x = True

        err_y -= abs_y
        if err_y < 0:
            err_y += n_ticks
            pulse_y = True

        # Raise pulses on both axes at once
        if pulse_x:
            GPIO.output(X_PUL, GPIO.HIGH)
        if pulse_y:
            GPIO.output(Y_PUL, GPIO.HIGH)

        t += half_period
        while perf_counter() < t:
            pass

        # Lower pulses on both axes at once
        if pulse_x:
            GPIO.output(X_PUL, GPIO.LOW)
        if pulse_y:
            GPIO.output(Y_PUL, GPIO.LOW)

        t += half_period
        while perf_counter() < t:
            pass


def move_to(
    pos: PointMM,
    target: PointMM,
    speed_mm_s: float,
) -> PointMM:
    dx_steps = round((target[0] - pos[0]) * STEPS_PER_MM)
    dy_steps = round((target[1] - pos[1]) * STEPS_PER_MM)
    move_steps(dx_steps, dy_steps, speed_mm_s)
    # Track position based on actual steps taken (avoids rounding drift)
    return (
        pos[0] + dx_steps / STEPS_PER_MM,
        pos[1] + dy_steps / STEPS_PER_MM,
    )


def load_paths(json_path: str) -> List[List[PointMM]]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    paths: List[List[PointMM]] = []
    for raw_path in data.get("paths", []):
        path = [(float(pt["x_mm"]), float(pt["y_mm"])) for pt in raw_path]
        if len(path) >= 2:
            paths.append(path)
    return paths


def main() -> None:
    paths = load_paths(PATHS_JSON)
    if not paths:
        print("No paths found in", PATHS_JSON)
        sys.exit(1)

    print(f"Loaded {len(paths)} paths.")
    print(f"Steps/mm: {STEPS_PER_MM}  Draw: {DRAW_SPEED_MM_S} mm/s  Travel: {TRAVEL_SPEED_MM_S} mm/s")
    input("\nPress Enter to start drawing (pen should be UP and carriage at origin)...")

    setup()
    motors_enable(True)
    pen_is_down = False
    pos: PointMM = (0.0, 0.0)

    try:
        for i, path in enumerate(paths):
            print(f"  Path {i + 1}/{len(paths)}  ({len(path)} pts)", end="\r")

            # Lift pen and travel to the start of this path
            if pen_is_down:
                pen_up()
                pen_is_down = False
            pos = move_to(pos, path[0], TRAVEL_SPEED_MM_S)

            # Drop pen and draw each segment
            pen_down()
            pen_is_down = True
            for pt in path[1:]:
                pos = move_to(pos, pt, DRAW_SPEED_MM_S)

        # Done — lift pen and return to origin
        if pen_is_down:
            pen_up()
        print(f"\nReturning to origin from {pos[0]:.1f}, {pos[1]:.1f} mm ...")
        pos = move_to(pos, (0.0, 0.0), TRAVEL_SPEED_MM_S)
        pen_down()  # de-energise solenoid when done
        print("Done!")

    except KeyboardInterrupt:
        print("\n[!] Interrupted — lifting pen.")
        if pen_is_down:
            pen_up()

    finally:
        motors_enable(False)
        GPIO.cleanup()


if __name__ == "__main__":
    main()
