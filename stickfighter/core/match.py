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
            ArrowHit(
                owner=owner, target=target_idx, part=part, damage=damage, x=point[0], y=point[1]
            )
        )
        coins = coins_for_part(part)
        self.archers[owner].coins += coins
        events.append(CoinEarned(who=owner, amount=coins))
        if target.hp <= 0.0 and target.alive:
            target.alive = False
            self.archers[owner].kills += 1
            events.append(Death(who=target_idx, killer=owner, headshot=(part == "head")))

    def _out_of_bounds(self, arrow: ArrowState) -> bool:
        m = C.OUT_OF_BOUNDS_MARGIN
        return (
            arrow.x < -m
            or arrow.x > C.ARENA_WIDTH + m
            or arrow.y > C.ARENA_HEIGHT + m
            or arrow.y < -m
        )

    def _maybe_end_round(self, events: list[Event]) -> None:
        p0_alive, p1_alive = self.archers[0].alive, self.archers[1].alive
        ended = False
        winner: int | None = None
        if not p0_alive or not p1_alive:
            ended = True
            if p0_alive:
                winner = 0
            elif p1_alive:
                winner = 1
        elif self.round_time_left <= 0.0:
            ended = True
            hp0, hp1 = self.archers[0].hp, self.archers[1].hp
            if hp0 > hp1:
                winner = 0
            elif hp1 > hp0:
                winner = 1
        if not ended:
            return
        if winner == 0:
            self.p0_rounds += 1
        elif winner == 1:
            self.p1_rounds += 1
        events.append(RoundOver(winner=winner, p0_rounds=self.p0_rounds, p1_rounds=self.p1_rounds))
        match_winner = is_match_won(self.p0_rounds, self.p1_rounds)
        if match_winner is not None:
            self.match_over = True
            self.winner = match_winner
            events.append(MatchOver(winner=match_winner))
        else:
            self._start_round()

    def snapshot(self) -> MatchSnapshot:
        return MatchSnapshot(
            arena=self.arena,
            archers=(self.archers[0], self.archers[1]),
            arrows=tuple(self.arrows),
            wind=self.wind,
            round_time_left=self.round_time_left,
            p0_rounds=self.p0_rounds,
            p1_rounds=self.p1_rounds,
            match_over=self.match_over,
            winner=self.winner,
        )

    def state_tuple(self) -> tuple:
        """A fully deterministic snapshot for equality comparisons in
        determinism tests -- not for rendering or observation encoding."""

        def archer_tuple(a: ArcherState) -> tuple:
            return (
                a.x, a.y, a.vx, a.vy, a.facing, a.grounded, a.hp, a.reload_timer,
                a.alive, a.aim_angle, a.draw, a.kills, a.coins,
            )

        def arrow_tuple(ar: ArrowState) -> tuple:
            return (ar.id, ar.x, ar.y, ar.vx, ar.vy, ar.owner, ar.age, ar.alive)

        return (
            tuple(archer_tuple(a) for a in self.archers),
            tuple(arrow_tuple(a) for a in self.arrows),
            self.wind,
            self.round_time_left,
            self.p0_rounds,
            self.p1_rounds,
            self.match_over,
            self.winner,
        )


def _selftest() -> None:
    """Plays a full headless match with random actions, then replays the same
    seed + action tape a second time and checks the two runs match exactly."""
    rng = random.Random(1234)
    tape: list[tuple[Action, Action]] = []

    def random_action(r: random.Random) -> Action:
        return Action(
            move=r.uniform(-1, 1),
            jump=r.random() < 0.05,
            aim_angle=r.uniform(-3.14159, 3.14159),
            draw=r.random(),
            release=r.random() < 0.1,
        )

    sim = MatchSim()
    sim.reset(seed=42)
    steps = 0
    max_steps = 20_000
    while not sim.match_over and steps < max_steps:
        a0, a1 = random_action(rng), random_action(rng)
        tape.append((a0, a1))
        sim.step(a0, a1)
        steps += 1
    print(
        f"match finished in {steps} steps, winner={sim.winner}, "
        f"score={sim.p0_rounds}-{sim.p1_rounds}"
    )

    replay = MatchSim()
    replay.reset(seed=42)
    for a0, a1 in tape:
        replay.step(a0, a1)

    if sim.state_tuple() == replay.state_tuple():
        print("determinism check: PASS")
    else:
        print("determinism check: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
    else:
        parser.print_help()
