import pytest

from stickfighter.ai.reward import (
    DAMAGE_SCALE,
    HEADSHOT_BONUS,
    NEAR_MISS_MAX_BONUS,
    ROUND_LOSS_PENALTY,
    ROUND_WIN_REWARD,
    STEP_PENALTY,
    compute_reward,
)
from stickfighter.core.arena import ARENAS
from stickfighter.core.entities import ArrowState
from stickfighter.core.events import ArrowHit, Death, RoundOver
from stickfighter.core.match import MatchSim, MatchSnapshot

_FLAT = next(a for a in ARENAS if a.name == "flat")


def _base_snapshot() -> MatchSnapshot:
    sim = MatchSim()
    sim.reset(seed=0, arena=_FLAT)
    return sim.snapshot()
