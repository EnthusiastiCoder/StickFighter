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
