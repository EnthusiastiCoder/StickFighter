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

    rel_x = mx(enemy_a.x) - mx(self_a.x)
    rel_y = enemy_a.y - self_a.y
    feats += [
        rel_x / C.ARENA_WIDTH,
        rel_y / C.ARENA_HEIGHT,
        mvx(enemy_a.vx) / CHAR_VEL_NORM,
        enemy_a.vy / CHAR_VEL_NORM,
        1.0 if enemy_a.grounded else -1.0,
        enemy_a.hp / C.CHAR_MAX_HP,
        enemy_a.draw,
        enemy_a.reload_timer / C.RELOAD_TIME,
        math.sin(enemy_a.aim_angle),
        -math.cos(enemy_a.aim_angle) if mirror else math.cos(enemy_a.aim_angle),
    ]

    nearest_arrows = sorted(
        snapshot.arrows, key=lambda a: (a.x - self_a.x) ** 2 + (a.y - self_a.y) ** 2
    )[:MAX_ARROWS]
    for i in range(MAX_ARROWS):
        if i < len(nearest_arrows):
            arrow = nearest_arrows[i]
            feats += [
                (mx(arrow.x) - mx(self_a.x)) / C.ARENA_WIDTH,
                (arrow.y - self_a.y) / C.ARENA_HEIGHT,
                mvx(arrow.vx) / ARROW_VEL_NORM,
                arrow.vy / ARROW_VEL_NORM,
                1.0 if arrow.owner == self_idx else -1.0,
                1.0,
            ]
        else:
            feats += [0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

    floating_platforms = snapshot.arena.platforms[1:][:MAX_PLATFORMS]  # [0] is always the ground
    for i in range(MAX_PLATFORMS):
        if i < len(floating_platforms):
            p = floating_platforms[i]
            center_x = p.x + p.width / 2.0
            feats += [
                (mx(center_x) - mx(self_a.x)) / C.ARENA_WIDTH,
                (p.y - self_a.y) / C.ARENA_HEIGHT,
                (p.width / 2.0) / C.ARENA_WIDTH,
                1.0,
            ]
        else:
            feats += [0.0, 0.0, 0.0, -1.0]

    feats.append((-snapshot.wind if mirror else snapshot.wind) / C.WIND_MAX)
    feats.append(snapshot.round_time_left / C.ROUND_TIME_LIMIT)
    self_rounds = snapshot.p0_rounds if self_idx == 0 else snapshot.p1_rounds
    enemy_rounds = snapshot.p1_rounds if self_idx == 0 else snapshot.p0_rounds
    feats.append((self_rounds - enemy_rounds) / C.WIN_ROUNDS)
