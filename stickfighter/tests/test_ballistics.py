import math

import pytest

from stickfighter.core import constants as C
from stickfighter.core.ballistics import (
    segment_hits_circle,
    segment_hits_rect,
    solve_launch_angle,
)
from stickfighter.core.entities import Circle, Rect


def _simulate_landing(angle, speed, gravity, wind, dx_target, dt=1.0 / 240, steps=20_000):
    """Forward-integrates a projectile and returns the point closest to
    x == dx_target, used to check solve_launch_angle's inverse solution."""
    x = y = 0.0
    vx, vy = math.cos(angle) * speed, math.sin(angle) * speed
    best = (x, y)
    for _ in range(steps):
        vy += gravity * dt
        vx += wind * dt
        x += vx * dt
        y += vy * dt
        if abs(x - dx_target) < abs(best[0] - dx_target):
            best = (x, y)
    return best
