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


def test_step_penalty_applies_with_no_events():
    assert compute_reward([], _base_snapshot(), 0) == pytest.approx(STEP_PENALTY)


def test_headshot_dealt_gives_bonus_plus_round_win_no_damage_scale():
    snap = _base_snapshot()
    events = [
        ArrowHit(owner=0, target=1, part="head", damage=9999.0, x=0, y=0),
        Death(who=1, killer=0, headshot=True, x=0, y=0, facing=1),
        RoundOver(winner=0, p0_rounds=1, p1_rounds=0),
    ]
    reward = compute_reward(events, snap, perspective=0)
    expected = STEP_PENALTY + HEADSHOT_BONUS + ROUND_WIN_REWARD
    assert reward == pytest.approx(expected)
