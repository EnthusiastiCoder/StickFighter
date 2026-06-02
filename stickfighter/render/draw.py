"""Stick-figure and bow/arrow rendering. Pure drawing: reads ArcherState /
ArrowState, never mutates them."""

import math

import pygame

from stickfighter.core import constants as C
from stickfighter.core.arena import Arena
from stickfighter.core.ballistics import bow_origin
from stickfighter.core.entities import ArcherState, ArrowState
from stickfighter.render import theme

PLATFORM_THICKNESS = 14.0

LINE_WIDTH = 5
HEAD_RADIUS = C.HEAD_RADIUS + 1

BOW_HALF_LEN = 20.0
BOW_CURVE_BASE = 4.0
BOW_CURVE_DRAW = 10.0
STRING_PULL_BASE = 6.0
STRING_PULL_MAX = 22.0

WALK_FREQUENCY = 9.0
WALK_SWING = 13.0


def _aim_vectors(angle: float) -> tuple[tuple[float, float], tuple[float, float]]:
    forward = (math.cos(angle), math.sin(angle))
    perp = (-forward[1], forward[0])
    return forward, perp


def _quad_bezier(p0, p1, p2, segments: int = 10) -> list[tuple[float, float]]:
    pts = []
    for i in range(segments + 1):
        t = i / segments
        mt = 1.0 - t
        x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
        y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
        pts.append((x, y))
    return pts


def _leg_points(
    hip: tuple[float, float], feet_y: float, x_offset: float, bend: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    foot = (hip[0] + x_offset, feet_y)
    knee = ((hip[0] + foot[0]) / 2.0 + bend, (hip[1] + feet_y) / 2.0)
    return hip, knee, foot


def draw_archer(surface: pygame.Surface, archer: ArcherState, elapsed: float) -> None:
    if not archer.alive:
        return
    top = archer.top()
    head = (archer.x, top + C.HEAD_CENTER_Y)
    shoulder = (archer.x, top + C.TORSO_TOP_Y + 4.0)
    hip = (archer.x, top + C.TORSO_BOTTOM_Y)
    feet_y = archer.y

    _draw_legs(surface, archer, hip, feet_y, elapsed)
    _draw_bow_and_arms(surface, archer, shoulder)

    pygame.draw.line(surface, theme.INK, shoulder, hip, LINE_WIDTH)
    pygame.draw.circle(surface, theme.INK, (round(head[0]), round(head[1])), round(HEAD_RADIUS))


def _draw_legs(surface, archer: ArcherState, hip, feet_y: float, elapsed: float) -> None:
    speed_frac = min(abs(archer.vx) / C.CHAR_MOVE_SPEED, 1.0)
    if not archer.grounded:
        legs = (
            _leg_points(hip, feet_y, x_offset=7.0 * archer.facing, bend=4.0 * archer.facing),
            _leg_points(hip, feet_y, x_offset=-9.0 * archer.facing, bend=-5.0 * archer.facing),
        )
    elif speed_frac > 0.05:
        phase = elapsed * WALK_FREQUENCY
        swing = math.sin(phase) * WALK_SWING * speed_frac
        legs = (
            _leg_points(hip, feet_y, x_offset=swing, bend=swing * 0.4),
            _leg_points(hip, feet_y, x_offset=-swing, bend=-swing * 0.4),
        )
    else:
        legs = (
            _leg_points(hip, feet_y, x_offset=6.0, bend=2.0),
            _leg_points(hip, feet_y, x_offset=-6.0, bend=-2.0),
        )
    for a, b, c in legs:
        pygame.draw.line(surface, theme.INK, a, b, LINE_WIDTH)
        pygame.draw.line(surface, theme.INK, b, c, LINE_WIDTH)
