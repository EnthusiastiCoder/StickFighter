"""Encodes a MatchSnapshot into a compact, egocentric observation vector.

The observation is always from the acting archer's point of view and always
mirrored so archer 1 sees a canonical "I am on the left, enemy is to my
right" frame -- the same policy can then play either side, doubling the
effective self-play data per match.
"""

import math

import numpy as np

from stickfighter.core import constants as C
from stickfighter.core.match import MatchSnapshot

MAX_ARROWS = C.MAX_ARROWS_TRACKED
MAX_PLATFORMS = 4
CHAR_VEL_NORM = C.CHAR_JUMP_SPEED
ARROW_VEL_NORM = C.ARROW_SPEED_MAX

_SELF_DIM = 10
_ENEMY_DIM = 10
_ARROW_DIM = 6
_PLATFORM_DIM = 4
OBS_DIM = _SELF_DIM + _ENEMY_DIM + MAX_ARROWS * _ARROW_DIM + MAX_PLATFORMS * _PLATFORM_DIM + 3


def encode_obs(snapshot: MatchSnapshot, perspective: int) -> np.ndarray:
    self_idx = perspective
    enemy_idx = 1 - perspective
    self_a = snapshot.archers[self_idx]
    enemy_a = snapshot.archers[enemy_idx]
    mirror = perspective == 1

    def mx(v: float) -> float:
        return C.ARENA_WIDTH - v if mirror else v

    def mvx(v: float) -> float:
        return -v if mirror else v

    feats: list[float] = []

    feats += [
        (mx(self_a.x) / C.ARENA_WIDTH) * 2.0 - 1.0,
        self_a.y / C.ARENA_HEIGHT,
        mvx(self_a.vx) / CHAR_VEL_NORM,
        self_a.vy / CHAR_VEL_NORM,
        1.0 if self_a.grounded else -1.0,
        self_a.hp / C.CHAR_MAX_HP,
        self_a.draw,
        self_a.reload_timer / C.RELOAD_TIME,
        math.sin(self_a.aim_angle),
        -math.cos(self_a.aim_angle) if mirror else math.cos(self_a.aim_angle),
    ]
