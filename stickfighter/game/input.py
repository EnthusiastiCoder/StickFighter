"""Human input: WASD/arrow movement + mouse drag-to-aim, decoded into an
Action every frame. This is the only place mouse/keyboard state is read --
everything downstream, including the trained AI, only ever sees an Action.
"""

import math

import pygame

from stickfighter.core.actions import Action

MAX_DRAG_PIXELS = 140.0


class HumanInputController:
    def __init__(self) -> None:
        self._drawing = False
        self._release_pending = False

    def handle_pygame_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._drawing = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._drawing:
                self._release_pending = True
            self._drawing = False

    def build_action(self, anchor: tuple[float, float]) -> Action:
        keys = pygame.key.get_pressed()
        move = 0.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            move -= 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            move += 1.0
        jump = bool(keys[pygame.K_w] or keys[pygame.K_UP] or keys[pygame.K_SPACE])

        mouse_x, mouse_y = pygame.mouse.get_pos()
        dx, dy = mouse_x - anchor[0], mouse_y - anchor[1]
        dist = math.hypot(dx, dy)
        aim_angle = math.atan2(dy, dx) if dist > 1e-3 else 0.0

        is_active_draw = self._drawing or self._release_pending
        draw = min(dist / MAX_DRAG_PIXELS, 1.0) if is_active_draw else 0.0

        release = self._release_pending
        self._release_pending = False
        return Action(move=move, jump=jump, aim_angle=aim_angle, draw=draw, release=release)
