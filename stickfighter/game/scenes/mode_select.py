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


class ModeSelectScene:
    def __init__(self) -> None:
        self._next_scene: DuelScene | None = None
        width = C.ARENA_WIDTH
        button_w, button_h, gap = 180.0, 70.0, 24.0
        names = list(DIFFICULTIES)
        total_w = len(names) * button_w + (len(names) - 1) * gap
        start_x = width / 2 - total_w / 2
        y = C.ARENA_HEIGHT / 2 - 60

        self._difficulty_buttons = []
        for i, name in enumerate(names):
            x = round(start_x + i * (button_w + gap))
            rect = pygame.Rect(x, round(y), round(button_w), round(button_h))
            self._difficulty_buttons.append((rect, name))

        spectate_y = round(y + button_h + 40)
        self._spectate_button = pygame.Rect(round(width / 2 - 110), spectate_y, 220, 60)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        pos = event.pos
        for rect, name in self._difficulty_buttons:
            if rect.collidepoint(pos):
                self._next_scene = DuelScene(opponent_factory=DIFFICULTIES[name])
                return
        if self._spectate_button.collidepoint(pos):
            self._next_scene = DuelScene(opponent_factory=make_legend, player0_factory=make_gold)

    def update(self, dt: float) -> "DuelScene | None":
        return self._next_scene

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(theme.BG)
        title = _font(40).render("CHOOSE YOUR OPPONENT", True, theme.INK)
        surface.blit(title, (surface.get_width() / 2 - title.get_width() / 2, 70))

        mouse_pos = pygame.mouse.get_pos()
        for rect, name in self._difficulty_buttons:
            self._draw_button(surface, rect, name, hovered=rect.collidepoint(mouse_pos))
        spectate_hovered = self._spectate_button.collidepoint(mouse_pos)
        self._draw_button(surface, self._spectate_button, "SPECTATE AI", hovered=spectate_hovered)
