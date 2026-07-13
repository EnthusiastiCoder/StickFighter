"""Mode/difficulty select: choose a duel difficulty, or watch two AIs fight."""

import pygame

from stickfighter.core import constants as C
from stickfighter.game.difficulty import DIFFICULTIES, make_gold, make_legend
from stickfighter.game.scenes.duel import DuelScene
from stickfighter.render import theme

_FONT_CACHE: dict[int, pygame.font.Font] = {}


def _font(size: int) -> pygame.font.Font:
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = pygame.font.Font(None, size)
    return _FONT_CACHE[size]
