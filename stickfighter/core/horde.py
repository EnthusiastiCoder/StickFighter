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
