"""Rendering for horde-mode melee enemies. Reuses draw.py's leg-cycle helper
(melee enemies walk on the same ground plane as an archer) but replaces the
bow/arms with a weapon read at a glance: a short blade for swordsmen, a long
line for spearmen's reach, and a shield block in front of shieldbearers.
"""

import pygame

from stickfighter.core import constants as C
from stickfighter.core.horde_entities import MeleeEnemyState
from stickfighter.render import theme
from stickfighter.render.draw import (
    HEAD_RADIUS,
    LINE_WIDTH,
    VISUAL_HIP_Y,
    _circle,
    _draw_legs,
    _line,
)

SWORD_LENGTH = 22.0
SPEAR_LENGTH = 38.0
SHIELD_WIDTH = 10.0
SHIELD_HEIGHT = 34.0


def draw_melee_enemy(
    surface: pygame.Surface, enemy: MeleeEnemyState, elapsed: float, scale: float = 1.0
) -> None:
    if not enemy.alive:
        return
    top = enemy.top()
    head = (enemy.x, top + C.HEAD_CENTER_Y)
    shoulder = (enemy.x, top + C.TORSO_TOP_Y + 4.0)
    hip = (enemy.x, top + VISUAL_HIP_Y)
    feet_y = enemy.y

    _draw_legs(surface, enemy, hip, feet_y, elapsed, scale)
    _draw_weapon(surface, enemy, shoulder, scale)

    _line(surface, scale, theme.INK, shoulder, hip, LINE_WIDTH)
    _circle(surface, scale, head, HEAD_RADIUS, theme.INK)

    if enemy.config.has_shield:
        _draw_shield(surface, enemy, shoulder, hip, scale)


def _draw_weapon(
    surface: pygame.Surface, enemy: MeleeEnemyState, shoulder: tuple[float, float], scale: float
) -> None:
    length = SPEAR_LENGTH if enemy.config.name == "spearman" else SWORD_LENGTH
    hand = (shoulder[0] + enemy.facing * 8.0, shoulder[1] + 6.0)
    tip = (hand[0] + enemy.facing * length, hand[1] - 4.0)
    _line(surface, scale, theme.INK, shoulder, hand, LINE_WIDTH)
    _line(surface, scale, theme.INK, hand, tip, 3)


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
