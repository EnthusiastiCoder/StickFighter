"""The playable duel: human archer (P0) vs a chosen difficulty's opponent."""

from collections.abc import Callable

import pygame

from stickfighter.ai.bots.base import Policy
from stickfighter.core import constants as C
from stickfighter.core.events import CoinEarned, Event
from stickfighter.core.match import MatchSim
from stickfighter.game import save
from stickfighter.game.difficulty import make_silver
from stickfighter.game.input import HumanInputController
from stickfighter.render import draw, effects, hud, ragdoll, theme

_FONT_CACHE: dict[int, pygame.font.Font] = {}


def _font(size: int) -> pygame.font.Font:
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = pygame.font.Font(None, size)
    return _FONT_CACHE[size]


class DuelScene:
    """Also doubles as the AI-vs-AI spectate scene: pass a `player0_factory`
    to replace the human with a bot, instead of duplicating this entire
    render/effects/ragdoll pipeline in a second class."""

    def __init__(
        self,
        opponent_factory: Callable[[], Policy] = make_silver,
        player0_factory: Callable[[], Policy] | None = None,
        seed: int | None = None,
    ) -> None:
        self.save_data = save.load()
        self.sim = MatchSim()
        self.opponent = opponent_factory()
        self.player0 = player0_factory() if player0_factory is not None else None
        self.input = HumanInputController() if self.player0 is None else None
        self._new_match(seed)
        self.effects = effects.EffectsSystem(gore_enabled=self.save_data.gore_enabled)
        self.ragdolls = ragdoll.RagdollManager()
        self.elapsed = 0.0
        self.paused = False
        self._scene_surface = pygame.Surface((int(C.ARENA_WIDTH), int(C.ARENA_HEIGHT)))
        self._snapshot = self.sim.snapshot()

    def _new_match(self, seed: int | None = None) -> None:
        self.sim.reset(seed=seed)
        self.opponent.reset(seed=seed)
        if self.player0 is not None:
            self.player0.reset(seed=seed)

    def handle_event(self, event: pygame.event.Event) -> None:
        if self.input is not None:
            self.input.handle_pygame_event(event)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.paused = not self.paused
            elif event.key == pygame.K_RETURN and self.sim.match_over:
                self._new_match()
            elif event.key == pygame.K_g:
                self.save_data.gore_enabled = not self.save_data.gore_enabled
                self.effects.gore_enabled = self.save_data.gore_enabled
                save.save(self.save_data)

    def update(self, dt: float) -> None:
        if self.paused:
            return
        self.elapsed += dt

        if self.player0 is not None:
            action0 = self.player0.act(self.sim.snapshot(), 0)
        else:
            archer0 = self.sim.archers[0]
            anchor = (archer0.x, archer0.top() + C.BOW_HEIGHT_OFFSET)
            action0 = self.input.build_action(anchor)
        action1 = self.opponent.act(self.sim.snapshot(), 1)

        events = self.sim.step(action0, action1)
        self._snapshot = self.sim.snapshot()
        if self.player0 is None:  # only a real human's coins carry over
            self._accumulate_coins(events)
        self.effects.handle_events(events)
        self.ragdolls.handle_events(events)
        self.effects.update(dt, self._snapshot.arrows)
        self.ragdolls.update(dt)

    def _accumulate_coins(self, events: list[Event]) -> None:
        earned = sum(e.amount for e in events if isinstance(e, CoinEarned) and e.who == 0)
        if earned:
            self.save_data.total_coins += earned
            save.save(self.save_data)

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

    def _draw_banner(self, surface: pygame.Surface, title: str, subtitle: str) -> None:
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        overlay.fill((*theme.INK, 140))
        surface.blit(overlay, (0, 0))

        title_surf = _font(56).render(title, True, theme.WHITE)
        sub_surf = _font(28).render(subtitle, True, theme.WHITE)
        cx, cy = surface.get_width() / 2, surface.get_height() / 2
        surface.blit(title_surf, (cx - title_surf.get_width() / 2, cy - 40))
        surface.blit(sub_surf, (cx - sub_surf.get_width() / 2, cy + 20))
