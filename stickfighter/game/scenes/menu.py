"""Minimal title screen: click or press any key to start a duel."""

import math

import pygame

from stickfighter.core import constants as C
from stickfighter.game.scenes.mode_select import ModeSelectScene
from stickfighter.render import theme

_FONT_CACHE: dict[int, pygame.font.Font] = {}


def _font(size: int) -> pygame.font.Font:
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = pygame.font.Font(None, size)
    return _FONT_CACHE[size]


class MenuScene:
    def __init__(self) -> None:
        self.elapsed = 0.0
        self._start_requested = False

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
            self._start_requested = True

    def update(self, dt: float) -> "ModeSelectScene | None":
        self.elapsed += dt
        return ModeSelectScene() if self._start_requested else None

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(theme.BG)
        cx = C.ARENA_WIDTH / 2

        title = _font(72).render("STICKFIGHTER", True, theme.INK)
        surface.blit(title, (cx - title.get_width() / 2, C.ARENA_HEIGHT / 2 - 80))

        prompt = _font(28).render("Click or press any key to continue", True, theme.INK)
        prompt.set_alpha(128 + round(127 * math.sin(self.elapsed * 3.0)))
        surface.blit(prompt, (cx - prompt.get_width() / 2, C.ARENA_HEIGHT / 2 + 20))
