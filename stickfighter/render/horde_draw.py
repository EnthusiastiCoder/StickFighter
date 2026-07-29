"""Rendering for horde-mode melee enemies. Reuses draw.py's leg-cycle helper
(melee enemies walk on the same ground plane as an archer) but replaces the
bow/arms with a weapon read at a glance: a short blade for swordsmen, a long
line for spearmen's reach, and a shield block in front of shieldbearers.

Weapon-arm swing and the forward lean follow standard run-cycle principles
(researched, not guessed): the arm swings opposite the leading leg and
swings higher/faster than a walk; the whole torso leans into the direction
of travel, scaling with effort, while the head stays aligned with the spine.
"""

import math

import pygame

from stickfighter.core import constants as C
from stickfighter.core.horde_entities import MeleeEnemyState
from stickfighter.render import theme
from stickfighter.render.draw import (
    HEAD_RADIUS,
    LINE_WIDTH,
    VISUAL_HIP_Y,
    WALK_FREQUENCY,
    WALK_SWING,
    _circle,
    _draw_legs,
    _line,
)

SWORD_LENGTH = 18.0
SPEAR_LENGTH = 34.0
SHIELD_WIDTH = 10.0
SHIELD_HEIGHT = 34.0
ARM_SWING = WALK_SWING * 1.3  # runs swing arms higher/faster than a walk
LEAN_MAX = 9.0  # forward torso lean at full speed, pivoting from the hip
HAND_RADIUS = 2.5


def draw_melee_enemy(
    surface: pygame.Surface, enemy: MeleeEnemyState, elapsed: float, scale: float = 1.0
) -> None:
    if not enemy.alive:
        return
    top = enemy.top()
    hip = (enemy.x, top + VISUAL_HIP_Y)
    feet_y = enemy.y

    speed_frac = min(abs(enemy.vx) / C.CHAR_MOVE_SPEED, 1.0)
    # lean the torso into the run, pivoting from the hip -- peaks at full
    # speed, vanishes when stopped (e.g. mid melee-range attack)
    lean = -enemy.facing * LEAN_MAX * speed_frac
    head = (enemy.x + lean, top + C.HEAD_CENTER_Y)
    shoulder = (enemy.x + lean * 0.6, top + C.TORSO_TOP_Y + 4.0)

    phase = elapsed * WALK_FREQUENCY
    leg_swing = math.sin(phase) * WALK_SWING * speed_frac if speed_frac > 0.05 else 0.0

    _draw_legs(surface, enemy, hip, feet_y, elapsed, scale)
    _draw_weapon(surface, enemy, shoulder, leg_swing, speed_frac, scale)

    _line(surface, scale, theme.INK, shoulder, hip, LINE_WIDTH)
    _circle(surface, scale, head, HEAD_RADIUS, theme.INK)

    if enemy.config.has_shield:
        _draw_shield(surface, enemy, shoulder, hip, scale)


def _draw_weapon(
    surface: pygame.Surface,
    enemy: MeleeEnemyState,
    shoulder: tuple[float, float],
    leg_swing: float,
    speed_frac: float,
    scale: float,
) -> None:
    # opposite the leading leg (leg_swing>0 means the *other* leg leads),
    # and swings noticeably harder than the leg cycle -- per run-cycle
    # reference, arms move higher/faster than legs in a run, not a walk.
    arm_swing = -leg_swing * (ARM_SWING / WALK_SWING) if WALK_SWING else 0.0
    forward_lift = 6.0 + 6.0 * speed_frac

    hand = (
        shoulder[0] + enemy.facing * forward_lift + arm_swing,
        shoulder[1] + 10.0 - abs(arm_swing) * 0.3,
    )
    length = SPEAR_LENGTH if enemy.config.name == "spearman" else SWORD_LENGTH
    tip = (hand[0] + enemy.facing * length, hand[1] - 3.0)

    _line(surface, scale, theme.INK, shoulder, hand, LINE_WIDTH)
    _circle(surface, scale, hand, HAND_RADIUS, theme.INK)
    _line(surface, scale, theme.INK, hand, tip, 3)


CLOUD_PUFF_RADII = (16.0, 13.0, 11.0, 10.0)
CLOUD_PUFF_OFFSETS = ((-18.0, 2.0), (0.0, -4.0), (18.0, 2.0), (32.0, 8.0))


def draw_cloud(surface: pygame.Surface, x: float, y: float, scale: float) -> None:
    """A few overlapping circles beneath a cloud archer's feet -- enough to
    read as a cloud at a glance without needing a dedicated sprite."""
    for (dx, dy), r in zip(CLOUD_PUFF_OFFSETS, CLOUD_PUFF_RADII):
        _circle(surface, scale, (x + dx, y + dy), r, theme.WHITE)


def _draw_shield(
    surface: pygame.Surface,
    enemy: MeleeEnemyState,
    shoulder: tuple[float, float],
    hip: tuple[float, float],
    scale: float,
) -> None:
    cx = enemy.x + enemy.facing * (C.CHAR_HALF_WIDTH + SHIELD_WIDTH / 2.0)
    cy = (shoulder[1] + hip[1]) / 2.0
    w, h = round(SHIELD_WIDTH * scale), round(SHIELD_HEIGHT * scale)
    rect = pygame.Rect(0, 0, w, h)
    rect.center = (round(cx * scale), round(cy * scale))
    border = max(1, round(3 * scale))
    outline_width = max(1, round(2 * scale))
    pygame.draw.rect(surface, theme.INK, rect, border_radius=border)
    pygame.draw.rect(surface, theme.GOLD, rect, width=outline_width, border_radius=border)
