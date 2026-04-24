#!/usr/bin/env python3
"""
test_motors.py — Draw a simple circle to verify both motors move together correctly.
Run this on the Raspberry Pi before using draw_paths.py.
"""
import math
import sys
from time import perf_counter, sleep

import RPi.GPIO as GPIO

# ============================================================
#  CONFIGURATION — same as draw_paths.py, adjust to match
# ============================================================
STEPS_PER_MM      = 80.0   # calibrate this!
SPEED_MM_S        = 15.0   # drawing speed (mm/s)

X_PUL, X_DIR, X_ENA = 19, 13, 17
X_INVERT = False

Y_PUL, Y_DIR, Y_ENA = 12, 20, 21
Y_INVERT = False

PEN_PIN      = 16
PEN_PULSE_S  = 0.05
PEN_SETTLE_S = 0.3

# Shape parameters
CIRCLE_RADIUS_MM = 20.0    # radius of test circle in mm
CIRCLE_STEPS     = 72      # number of line segments (more = smoother)
# ============================================================


def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in (X_PUL, X_DIR, X_ENA, Y_PUL, Y_DIR, Y_ENA, PEN_PIN):
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)


def motors_enable(on):
    GPIO.output(X_ENA, GPIO.HIGH if on else GPIO.LOW)
    GPIO.output(Y_ENA, GPIO.HIGH if on else GPIO.LOW)


def pen_toggle():
    GPIO.output(PEN_PIN, GPIO.HIGH)
    sleep(PEN_PULSE_S)
    GPIO.output(PEN_PIN, GPIO.LOW)
    sleep(PEN_SETTLE_S)


def move_steps(dx, dy, speed_mm_s):
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

        if pulse_x:
            GPIO.output(X_PUL, GPIO.HIGH)
        if pulse_y:
            GPIO.output(Y_PUL, GPIO.HIGH)
        t += half_period
        while perf_counter() < t:
            pass
        if pulse_x:
            GPIO.output(X_PUL, GPIO.LOW)
        if pulse_y:
            GPIO.output(Y_PUL, GPIO.LOW)
        t += half_period
        while perf_counter() < t:
            pass


# Track position in mm so we accumulate integer steps correctly
_pos = [0.0, 0.0]

def move_to(tx, ty, speed_mm_s=None):
    if speed_mm_s is None:
        speed_mm_s = SPEED_MM_S
    dx_steps = round((tx - _pos[0]) * STEPS_PER_MM)
    dy_steps = round((ty - _pos[1]) * STEPS_PER_MM)
    move_steps(dx_steps, dy_steps, speed_mm_s)
    _pos[0] += dx_steps / STEPS_PER_MM
    _pos[1] += dy_steps / STEPS_PER_MM


def draw_circle(cx, cy, radius, n_segments):
    print(f"Drawing circle: center=({cx},{cy}) mm, radius={radius} mm, {n_segments} segments")
    # Travel to start point (rightmost point of circle)
    start_x = cx + radius
    start_y = cy
    move_to(start_x, start_y, speed_mm_s=SPEED_MM_S * 2)

    pen_toggle()  # pen down

    for i in range(1, n_segments + 1):
        angle = 2 * math.pi * i / n_segments
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        move_to(x, y)

    pen_toggle()  # pen up


def draw_square(cx, cy, side):
    half = side / 2
    corners = [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
        (cx - half, cy - half),  # close
    ]
    print(f"Drawing square: center=({cx},{cy}) mm, side={side} mm")
    move_to(*corners[0], speed_mm_s=SPEED_MM_S * 2)
    pen_toggle()  # pen down
    for corner in corners[1:]:
        move_to(*corner)
    pen_toggle()  # pen up


def main():
    print("Test shapes:")
    print("  1) Circle")
    print("  2) Square")
    print("  3) Both")
    choice = input("Choose (1/2/3): ").strip()

    setup()
    motors_enable(True)

    try:
        input("\nPress Enter to start (pen UP, carriage at origin)...")

        if choice in ("1", "3"):
            draw_circle(
                cx=CIRCLE_RADIUS_MM + 5,
                cy=CIRCLE_RADIUS_MM + 5,
                radius=CIRCLE_RADIUS_MM,
                n_segments=CIRCLE_STEPS,
            )

        if choice in ("2", "3"):
            offset = (CIRCLE_RADIUS_MM * 2 + 15) if choice == "3" else 0
            draw_square(
                cx=CIRCLE_RADIUS_MM + 5 + offset,
                cy=CIRCLE_RADIUS_MM + 5,
                side=CIRCLE_RADIUS_MM * 2,
            )

        print("\nReturning to origin...")
        move_to(0.0, 0.0, speed_mm_s=SPEED_MM_S * 2)
        print("Done!")

    except KeyboardInterrupt:
        print("\n[!] Interrupted.")

    finally:
        motors_enable(False)
        GPIO.cleanup()


if __name__ == "__main__":
    main()
