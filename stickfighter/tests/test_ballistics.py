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


@pytest.mark.parametrize(
    "dx,dy,speed,wind",
    [
        (300, -30, 700, 0.0),
        (300, -30, 700, 40.0),
        (-300, -30, 700, 0.0),
        (-300, 50, 700, 0.0),
        (300, 100, 700, 0.0),
        (150, -80, 900, 0.0),
        (600, -20, 950, 0.0),
    ],
)
def test_solve_launch_angle_round_trips_to_target(dx, dy, speed, wind):
    angle = solve_launch_angle(dx, dy, speed, C.GRAVITY, prefer_low=True)
    assert angle is not None
    x, y = _simulate_landing(angle, speed, C.GRAVITY, wind, dx)
    assert math.hypot(x - dx, y - dy) < 4.0


def test_solve_launch_angle_high_arc_also_round_trips():
    dx, dy, speed = 300, -30, 700
    angle = solve_launch_angle(dx, dy, speed, C.GRAVITY, prefer_low=False)
    assert angle is not None
    x, y = _simulate_landing(angle, speed, C.GRAVITY, 0.0, dx)
    assert math.hypot(x - dx, y - dy) < 4.0


def test_solve_launch_angle_out_of_range_returns_none():
    # Max flat range at this speed is speed^2/gravity; well beyond that is unreachable.
    max_range = 700**2 / C.GRAVITY
    assert solve_launch_angle(max_range * 3, 0.0, 700, C.GRAVITY) is None
