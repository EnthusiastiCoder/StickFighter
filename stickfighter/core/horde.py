"""HordeSim: a hardcoded hero archer vs. waves of hardcoded melee enemies.

Not a game mode and not RL -- a standalone deterministic scenario built to
render one highlight video. The hero is fully autonomous (PERFECT-tier
ballistic aim, no reaction delay), so there is no external Action input at
all: reset(seed) + repeated step() is the whole interface.

Reuses core/ballistics.py and core/rules.py unchanged, and emits the same
Event types core/match.py does (ArrowFired, ArrowHit, ArrowMiss, Death,
CoinEarned) so the existing render/effects.py EffectsSystem and
render/ragdoll.py RagdollManager work on this scenario with no changes --
both are already driven purely by event data, not by any "there are exactly
2 archers" assumption.
"""

import argparse
import math
import random
from dataclasses import dataclass, field

from stickfighter.core import constants as C
from stickfighter.core.ballistics import (
    segment_hits_circle,
    segment_hits_rect,
    solve_launch_angle,
    spawn_arrow,
    step_arrow,
)
from stickfighter.core.entities import (
    ArcherState,
    ArrowState,
    head_hitbox,
    limb_hitbox,
    torso_hitbox,
)
from stickfighter.core.events import ArrowFired, ArrowHit, ArrowMiss, CoinEarned, Death, Event
from stickfighter.core.horde_entities import SHIELDBEARER, SPEARMAN, SWORDSMAN, MeleeEnemyState
from stickfighter.core.rules import coins_for_part, damage_for_part

HERO_X = 90.0
TOTAL_ENEMIES = 15
MAX_ACTIVE = 3
ATTACK_COOLDOWN = 0.6
ENEMY_MIX = ((SWORDSMAN, 0.5), (SPEARMAN, 0.3), (SHIELDBEARER, 0.2))
SPAWN_X_JITTER = (20.0, 150.0)


@dataclass(frozen=True)
class HordeSnapshot:
    hero: ArcherState
    enemies: tuple[MeleeEnemyState, ...]
    arrows: tuple[ArrowState, ...]
    total_kills: int
    cleared: bool


def _weighted_choice(rng: random.Random, mix: tuple) -> object:
    total = sum(w for _cfg, w in mix)
    r = rng.uniform(0.0, total)
    upto = 0.0
    for cfg, w in mix:
        upto += w
        if r <= upto:
            return cfg
    return mix[-1][0]


@dataclass
class HordeSim:
    rng: random.Random = field(default_factory=random.Random)
    hero: ArcherState = field(default_factory=lambda: ArcherState(x=HERO_X, y=C.GROUND_Y))
    enemies: list[MeleeEnemyState] = field(default_factory=list)
    arrows: list[ArrowState] = field(default_factory=list)
    _queue: list = field(default_factory=list)
    _next_arrow_id: int = 0
    _next_enemy_id: int = 0
    total_kills: int = 0
    cleared: bool = False
    tick: int = 0
    log: list[dict] = field(default_factory=list)
    _claimed_targets: dict = field(default_factory=dict)  # arrow_id -> enemy_id already en route

    def reset(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.hero = ArcherState(x=HERO_X, y=C.GROUND_Y, facing=1, grounded=True)
        self.enemies = []
        self.arrows = []
        self._queue = [_weighted_choice(self.rng, ENEMY_MIX) for _ in range(TOTAL_ENEMIES)]
        self._next_arrow_id = 0
        self._next_enemy_id = 0
        self.total_kills = 0
        self.cleared = False
        self.tick = 0
        self.log = []
        self._claimed_targets = {}
        self._spawn_up_to_max()

    def _spawn_up_to_max(self) -> None:
        active = sum(1 for e in self.enemies if e.alive)
        while active < MAX_ACTIVE and self._queue:
            config = self._queue.pop(0)
            x = C.ARENA_WIDTH + self.rng.uniform(*SPAWN_X_JITTER)
            self.enemies.append(
                MeleeEnemyState(
                    enemy_id=self._next_enemy_id,
                    config=config,
                    x=x,
                    y=C.GROUND_Y,
                    hp=config.hp,
                    facing=-1,
                )
            )
            self._next_enemy_id += 1
            active += 1

    def step(self) -> list[Event]:
        events: list[Event] = []
        if self.cleared or not self.hero.alive:
            return events

        self._act_hero(events)
        self._update_enemies(events)
        self._update_arrows(events)
        self._spawn_up_to_max()

        if self.total_kills >= TOTAL_ENEMIES and not self._queue and not any(
            e.alive for e in self.enemies
        ):
            self.cleared = True
        self.tick += 1
        return events

    def _nearest_target(self) -> MeleeEnemyState | None:
        """Nearest *unclaimed* enemy (no arrow already flying toward it) when
        one exists, else nearest overall. Without this, the hero -- reload
        is much faster than an arrow's flight time -- would keep re-aiming
        at whichever enemy is nearest even after already firing at it,
        wasting a second arrow that arrives to find it already dead from the
        first. Confirmed from the per-shot log, not assumed: every miss in a
        first pass at this turned out to be exactly this case."""
        alive = [e for e in self.enemies if e.alive]
        if not alive:
            return None
        claimed_ids = set(self._claimed_targets.values())
        pool = [e for e in alive if e.enemy_id not in claimed_ids] or alive
        return min(pool, key=lambda e: abs(e.x - self.hero.x))
