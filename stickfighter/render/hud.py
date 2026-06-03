"""HUD: coins, round score, HP bars, a round timer, wind indicator, pause."""

import pygame

from stickfighter.core import constants as C
from stickfighter.core.match import MatchSnapshot
from stickfighter.render import theme

_font_cache: dict[int, pygame.font.Font] = {}


def _get_font(size: int) -> pygame.font.Font:
    if size not in _font_cache:
        _font_cache[size] = pygame.font.Font(None, size)
    return _font_cache[size]
