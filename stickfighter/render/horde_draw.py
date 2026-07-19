"""Rendering for horde-mode melee enemies. Reuses draw.py's leg-cycle helper
(melee enemies walk on the same ground plane as an archer) but replaces the
bow/arms with a weapon read at a glance: a short blade for swordsmen, a long
line for spearmen's reach, and a shield block in front of shieldbearers.
"""

import pygame

from stickfighter.core import constants as C
from stickfighter.core.horde_entities import MeleeEnemyState
from stickfighter.render import theme
from stickfighter.render.draw import HEAD_RADIUS, LINE_WIDTH, _draw_legs, draw_aa_circle

SWORD_LENGTH = 22.0
SPEAR_LENGTH = 38.0
SHIELD_WIDTH = 10.0
SHIELD_HEIGHT = 34.0


def draw_melee_enemy(surface: pygame.Surface, enemy: MeleeEnemyState, elapsed: float) -> None:
    if not enemy.alive:
        return
    top = enemy.top()
    head = (enemy.x, top + C.HEAD_CENTER_Y)
    shoulder = (enemy.x, top + C.TORSO_TOP_Y + 4.0)
    hip = (enemy.x, top + C.TORSO_BOTTOM_Y)
    feet_y = enemy.y
