"""Arrow spawning/integration, swept-segment collision tests, and the inverse
ballistics solver bots use to aim (see solve_launch_angle)."""

import math

from stickfighter.core import constants as C
from stickfighter.core.entities import ArcherState, ArrowState, Circle, Rect


def bow_origin(archer: ArcherState) -> tuple[float, float]:
    x = archer.x + archer.facing * C.BOW_FORWARD_OFFSET
    y = archer.top() + C.BOW_HEIGHT_OFFSET
    return x, y


def spawn_arrow(archer: ArcherState, owner: int, arrow_id: int) -> ArrowState:
    speed = C.ARROW_SPEED_MIN + archer.draw * (C.ARROW_SPEED_MAX - C.ARROW_SPEED_MIN)
    ox, oy = bow_origin(archer)
    vx = math.cos(archer.aim_angle) * speed
    vy = math.sin(archer.aim_angle) * speed
    return ArrowState(id=arrow_id, x=ox, y=oy, vx=vx, vy=vy, owner=owner)


def step_arrow(arrow: ArrowState, dt: float, wind: float) -> tuple[float, float]:
    """Integrates the arrow one step and returns its pre-step position, so the
    caller can run a swept-segment collision test against (prev -> new)."""
    prev = (arrow.x, arrow.y)
    arrow.vy += C.GRAVITY * dt
    arrow.vx += wind * dt
    arrow.x += arrow.vx * dt
    arrow.y += arrow.vy * dt
    arrow.age += dt
    return prev


def _closest_point_on_segment(
    p0: tuple[float, float], p1: tuple[float, float], q: tuple[float, float]
) -> tuple[float, float]:
    ax, ay = p0
    bx, by = p1
    qx, qy = q
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-9:
        return p0
    t = max(0.0, min(1.0, ((qx - ax) * dx + (qy - ay) * dy) / len_sq))
    return ax + t * dx, ay + t * dy


def segment_hits_circle(
    p0: tuple[float, float], p1: tuple[float, float], circle: Circle, extra_radius: float = 0.0
) -> tuple[float, float] | None:
    closest = _closest_point_on_segment(p0, p1, (circle.cx, circle.cy))
    r = circle.r + extra_radius
    if (closest[0] - circle.cx) ** 2 + (closest[1] - circle.cy) ** 2 <= r * r:
        return closest
    return None


def segment_hits_rect(
    p0: tuple[float, float], p1: tuple[float, float], rect: Rect, extra: float = 0.0
) -> tuple[float, float] | None:
    """Liang-Barsky segment-vs-AABB clipping; returns the entry point, or the
    segment start if it begins inside the (padded) rect."""
    x0, y0 = p0
    x1, y1 = p1
    left, right = rect.left - extra, rect.right + extra
    top, bottom = rect.top - extra, rect.bottom + extra
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - left), (dx, right - x0), (-dy, y0 - top), (dy, bottom - y0)):
        if p == 0.0:
            if q < 0.0:
                return None
            continue
        t = q / p
        if p < 0.0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)
    if t0 > t1:
        return None
    return x0 + t0 * dx, y0 + t0 * dy
