"""The playable duel: human archer (P0) vs a placeholder idle opponent (P1).

Real bots (stickfighter.ai.bots) replace the idle opponent once M3 lands --
this scene only needs *an* Action source for P1, not a specific one.
"""

import pygame

from stickfighter.core import constants as C
from stickfighter.core.actions import Action
from stickfighter.core.match import MatchSim
from stickfighter.game.input import HumanInputController
from stickfighter.render import draw, effects, hud, ragdoll, theme

NOOP = Action(move=0.0, jump=False, aim_angle=0.0, draw=0.0, release=False)

_FONT_CACHE: dict[int, pygame.font.Font] = {}


def _font(size: int) -> pygame.font.Font:
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = pygame.font.Font(None, size)
    return _FONT_CACHE[size]


class DuelScene:
    def __init__(self, seed: int | None = None) -> None:
        self.sim = MatchSim()
        self.sim.reset(seed=seed)
        self.input = HumanInputController()
        self.effects = effects.EffectsSystem()
        self.ragdolls = ragdoll.RagdollManager()
        self.elapsed = 0.0
        self.paused = False
        self._scene_surface = pygame.Surface((int(C.ARENA_WIDTH), int(C.ARENA_HEIGHT)))
        self._snapshot = self.sim.snapshot()

    def handle_event(self, event: pygame.event.Event) -> None:
        self.input.handle_pygame_event(event)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.paused = not self.paused
            elif event.key == pygame.K_RETURN and self.sim.match_over:
                self.sim.reset()

    def update(self, dt: float) -> None:
        if self.paused:
            return
        self.elapsed += dt

        archer0 = self.sim.archers[0]
        anchor = (archer0.x, archer0.top() + C.BOW_HEIGHT_OFFSET)
        action0 = self.input.build_action(anchor)
        action1 = NOOP

        events = self.sim.step(action0, action1)
        self._snapshot = self.sim.snapshot()
        self.effects.handle_events(events)
        self.ragdolls.handle_events(events)
        self.effects.update(dt, self._snapshot.arrows)
        self.ragdolls.update(dt)

    def draw(self, surface: pygame.Surface) -> None:
        snapshot = self._snapshot
        scene = self._scene_surface
        scene.fill(theme.BG)
        draw.draw_arena(scene, snapshot.arena)
        for archer in snapshot.archers:
            draw.draw_archer(scene, archer, self.elapsed)
        for arrow in snapshot.arrows:
            draw.draw_arrow(scene, arrow)
        self.effects.draw(scene)
        self.ragdolls.draw(scene)
        hud.draw_hud(scene, snapshot)

        surface.fill(theme.BG)
        surface.blit(scene, self.effects.screen_offset())

        if snapshot.match_over:
            self._draw_banner(surface, f"PLAYER {snapshot.winner + 1} WINS", "Enter to rematch")
        elif self.paused:
            self._draw_banner(surface, "PAUSED", "Esc to resume")
