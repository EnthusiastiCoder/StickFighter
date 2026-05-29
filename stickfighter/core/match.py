"""MatchSim: the deterministic, headless duel simulation.

This is the one piece of code that runs identically whether a human is
playing or the AI is training self-play -- there is no separate "fast"
physics model, so nothing can diverge between play and training.
"""

import argparse
import random
from dataclasses import dataclass

from stickfighter.core import constants as C
from stickfighter.core.actions import Action
from stickfighter.core.arena import Arena, random_arena
from stickfighter.core.ballistics import (
    segment_hits_circle,
    segment_hits_rect,
    spawn_arrow,
    step_arrow,
)
from stickfighter.core.character import step_character
from stickfighter.core.entities import (
    ArcherState,
    ArrowState,
    Rect,
    head_hitbox,
    limb_hitbox,
    torso_hitbox,
)
from stickfighter.core.events import (
    ArrowFired,
    ArrowHit,
    ArrowMiss,
    CoinEarned,
    Death,
    Event,
    MatchOver,
    RoundOver,
)
from stickfighter.core.rules import coins_for_part, damage_for_part, is_match_won


@dataclass(frozen=True)
class MatchSnapshot:
    """A read-only view for the renderer and observation encoder. The nested
    Arena/ArcherState/ArrowState objects are the sim's live instances, not
    copies -- callers must treat everything here as read-only."""

    arena: Arena
    archers: tuple[ArcherState, ArcherState]
    arrows: tuple[ArrowState, ...]
    wind: float
    round_time_left: float
    p0_rounds: int
    p1_rounds: int
    match_over: bool
    winner: int | None


class MatchSim:
    def __init__(self) -> None:
        self.rng = random.Random()
        self.arena: Arena = random_arena(self.rng)
        self.archers: list[ArcherState] = [ArcherState(x=0.0, y=0.0), ArcherState(x=0.0, y=0.0)]
        self.arrows: list[ArrowState] = []
        self.wind = 0.0
        self.round_time_left = C.ROUND_TIME_LIMIT
        self.p0_rounds = 0
        self.p1_rounds = 0
        self.match_over = False
        self.winner: int | None = None
        self._next_arrow_id = 0
        self.reset()

    def reset(self, seed: int | None = None, arena: Arena | None = None) -> None:
        self.rng = random.Random(seed)
        self.arena = arena if arena is not None else random_arena(self.rng)
        self.p0_rounds = 0
        self.p1_rounds = 0
        self.match_over = False
        self.winner = None
        self._next_arrow_id = 0
        self.archers = [ArcherState(x=0.0, y=0.0), ArcherState(x=0.0, y=0.0)]
        self._start_round()

    def _start_round(self) -> None:
        for archer, (x, y), facing in (
            (self.archers[0], self.arena.spawn_p0, 1),
            (self.archers[1], self.arena.spawn_p1, -1),
        ):
            archer.x, archer.y = x, y
            archer.vx = archer.vy = 0.0
            archer.facing = facing
            archer.grounded = True
            archer.hp = C.CHAR_MAX_HP
            archer.reload_timer = 0.0
            archer.alive = True
            archer.aim_angle = 0.0
            archer.draw = 0.0
        self.arrows = []
        self.wind = self.rng.uniform(-C.WIND_MAX, C.WIND_MAX)
        self.round_time_left = C.ROUND_TIME_LIMIT

    def step(self, a0: Action, a1: Action) -> list[Event]:
        events: list[Event] = []
        if self.match_over:
            return events

        actions = (a0, a1)
        for i, archer in enumerate(self.archers):
            step_character(archer, actions[i], self.arena, C.DT)
            if archer.alive:
                archer.reload_timer = max(0.0, archer.reload_timer - C.DT)
