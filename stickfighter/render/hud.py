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


def _draw_wind(surface: pygame.Surface, wind: float) -> None:
    cx, cy = C.ARENA_WIDTH / 2, 84.0
    length = min(abs(wind), C.WIND_MAX) / C.WIND_MAX * 30.0
    direction = 1.0 if wind >= 0.0 else -1.0
    end = (cx + direction * length, cy)
    pygame.draw.line(surface, theme.WIND_ARROW, (cx, cy), end, 3)
    pygame.draw.circle(surface, theme.WIND_ARROW, (round(end[0]), round(end[1])), 3)


def draw_hud(surface: pygame.Surface, snapshot: MatchSnapshot) -> None:
    font_big = _get_font(44)
    font_small = _get_font(26)
    width = C.ARENA_WIDTH

    pygame.draw.circle(surface, theme.GOLD, (26, 26), 10)
    pygame.draw.circle(surface, theme.INK, (26, 26), 10, 2)
    coin_text = font_small.render(str(snapshot.archers[0].coins), True, theme.WHITE)
    surface.blit(coin_text, (44, 14))

    score_text = font_big.render(f"{snapshot.p0_rounds} - {snapshot.p1_rounds}", True, theme.WHITE)
    surface.blit(score_text, (width / 2 - score_text.get_width() / 2, 6))

    timer_text = font_small.render(str(max(0, int(snapshot.round_time_left))), True, theme.WHITE)
    surface.blit(timer_text, (width / 2 - timer_text.get_width() / 2, 50))
