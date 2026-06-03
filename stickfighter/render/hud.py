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


def draw_hp_bar(
    surface: pygame.Surface, x: int, y: int, width: int, height: int, frac: float, flip: bool
) -> None:
    frac = max(0.0, min(1.0, frac))
    bg_rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(surface, theme.BAR_HP_BG, bg_rect)
    fill_w = round(width * frac)
    fg_x = x + width - fill_w if flip else x
    pygame.draw.rect(surface, theme.BAR_HP_FG, pygame.Rect(fg_x, y, fill_w, height))
    pygame.draw.rect(surface, theme.INK, bg_rect, 2)
