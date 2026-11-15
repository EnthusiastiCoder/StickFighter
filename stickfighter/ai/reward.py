"""Per-step reward shaping for one perspective.

Round outcomes (not raw deaths) carry the win/loss signal, since RoundOver
already correctly captures every way a round ends -- a kill, or a timeout
decided by HP -- with no risk of double-counting a kill as both a death
penalty and a separate win/loss term.

AIRBORNE_PENALTY and AIM_JITTER_PENALTY exist because a real trained
checkpoint was observed live: with nothing discouraging it, PPO found that
camping the arena wall while bunny-hopping nonstop and firing in wildly
swinging directions was an effective (if degenerate and ugly) strategy
against fixed-rule scripted bots. Both penalties are small relative to the
combat terms -- they nudge away from the specific exploit without
overwhelming the actual objective (hit the enemy, don't get hit, win).
"""

import math

from stickfighter.core import constants as C
from stickfighter.core.events import ArrowHit, Event, RoundOver
from stickfighter.core.match import MatchSnapshot

STEP_PENALTY = -0.001
ROUND_WIN_REWARD = 1.0
ROUND_LOSS_PENALTY = -1.0
DAMAGE_SCALE = 0.5 / 100.0  # +/-0.5 reward per 100 damage dealt/taken
HEADSHOT_BONUS = 0.15  # on top of the round win a kill already causes

NEAR_MISS_RADIUS = 300.0  # px; an own arrow within this of the enemy gets a shaping bonus
NEAR_MISS_MAX_BONUS = 0.02  # per step, at distance == 0 and scale == 1.0

AIRBORNE_PENALTY = 0.0008  # per step spent not grounded -- discourages nonstop bunny-hopping
# Scaled by |aim change| / pi. 0.03 (the original value) was strong enough
# that a real trained checkpoint found the reward-minimizing move was to
# saturate aim_angle at the actor's tanh boundary (exactly +/-pi) and never
# adjust it again -- cheaper than tracking the enemy -- rather than genuinely
# aiming well. Kept an order of magnitude smaller so it still damps noisy
# thrashing without outweighing the incentive to actually correct aim.
AIM_JITTER_PENALTY = 0.005


def compute_reward(
    events: list[Event],
    snapshot: MatchSnapshot,
    perspective: int,
    near_miss_scale: float = 0.0,
    prev_aim_angle: float | None = None,
) -> float:
    enemy = 1 - perspective
    reward = STEP_PENALTY
    self_archer = snapshot.archers[perspective]

    for e in events:
        if isinstance(e, ArrowHit):
            if e.part == "head":
                if e.target == enemy:
                    reward += HEADSHOT_BONUS
                # being headshot is already covered by the round-loss below
            elif e.target == enemy:
                reward += e.damage * DAMAGE_SCALE
            elif e.target == perspective:
                reward -= e.damage * DAMAGE_SCALE
        elif isinstance(e, RoundOver):
            if e.winner == perspective:
                reward += ROUND_WIN_REWARD
            elif e.winner == enemy:
                reward += ROUND_LOSS_PENALTY

    if near_miss_scale > 0.0:
        reward += _near_miss_bonus(snapshot, perspective, near_miss_scale)

    if not self_archer.grounded:
        reward -= AIRBORNE_PENALTY

    if prev_aim_angle is not None:
        delta = math.atan2(
            math.sin(self_archer.aim_angle - prev_aim_angle),
            math.cos(self_archer.aim_angle - prev_aim_angle),
        )
        reward -= AIM_JITTER_PENALTY * (abs(delta) / math.pi)

    return reward


def _near_miss_bonus(snapshot: MatchSnapshot, perspective: int, scale: float) -> float:
    enemy_a = snapshot.archers[1 - perspective]
    target = (enemy_a.x, enemy_a.top() + C.TORSO_TOP_Y)

    closest = math.inf
    for arrow in snapshot.arrows:
        if arrow.owner != perspective:
            continue
        d = math.hypot(arrow.x - target[0], arrow.y - target[1])
        closest = min(closest, d)

    if closest >= NEAR_MISS_RADIUS:
        return 0.0
    return scale * NEAR_MISS_MAX_BONUS * (1.0 - closest / NEAR_MISS_RADIUS)
