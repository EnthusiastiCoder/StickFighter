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

        for i, archer in enumerate(self.archers):
            if not archer.alive:
                continue
            if actions[i].release and archer.reload_timer <= 0.0 and archer.draw >= C.MIN_DRAW:
                arrow = spawn_arrow(archer, owner=i, arrow_id=self._next_arrow_id)
                self._next_arrow_id += 1
                self.arrows.append(arrow)
                archer.reload_timer = C.RELOAD_TIME
                archer.draw = 0.0
                events.append(ArrowFired(owner=i, x=arrow.x, y=arrow.y, vx=arrow.vx, vy=arrow.vy))

        surviving: list[ArrowState] = []
        for arrow in self.arrows:
            prev = step_arrow(arrow, C.DT, self.wind)
            if not self._resolve_arrow(arrow, prev, events):
                surviving.append(arrow)
        self.arrows = surviving

        self.round_time_left -= C.DT
        self._maybe_end_round(events)
        return events

    def _resolve_arrow(
        self, arrow: ArrowState, prev: tuple[float, float], events: list[Event]
    ) -> bool:
        """Returns True if the arrow is destroyed (hit or miss) this step."""
        cur = (arrow.x, arrow.y)
        target_idx = 1 - arrow.owner
        target = self.archers[target_idx]
        if target.alive:
            for part, hitbox, test in (
                ("head", head_hitbox(target), segment_hits_circle),
                ("torso", torso_hitbox(target), segment_hits_rect),
                ("limb", limb_hitbox(target), segment_hits_rect),
            ):
                point = test(prev, cur, hitbox, C.ARROW_COLLIDE_RADIUS)
                if point is not None:
                    self._apply_hit(arrow.owner, target_idx, part, point, events)
                    return True
        for platform in self.arena.platforms:
            rect = Rect(platform.x, platform.y, platform.right, platform.y + 6.0)
            point = segment_hits_rect(prev, cur, rect, C.ARROW_COLLIDE_RADIUS)
            if point is not None:
                events.append(ArrowMiss(owner=arrow.owner, x=point[0], y=point[1]))
                return True
        if self._out_of_bounds(arrow):
            events.append(ArrowMiss(owner=arrow.owner, x=arrow.x, y=arrow.y))
            return True
        return False

    def _apply_hit(
        self,
        owner: int,
        target_idx: int,
        part: str,
        point: tuple[float, float],
        events: list[Event],
    ) -> None:
        target = self.archers[target_idx]
        damage = damage_for_part(part)
        target.hp = max(0.0, target.hp - damage)
        events.append(
            ArrowHit(owner=owner, target=target_idx, part=part, damage=damage, x=point[0], y=point[1])
        )
        coins = coins_for_part(part)
        self.archers[owner].coins += coins
        events.append(CoinEarned(who=owner, amount=coins))
        if target.hp <= 0.0 and target.alive:
            target.alive = False
            self.archers[owner].kills += 1
            events.append(Death(who=target_idx, killer=owner, headshot=(part == "head")))
