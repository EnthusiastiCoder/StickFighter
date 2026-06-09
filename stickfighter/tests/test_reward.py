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


def test_headshot_taken_gives_only_round_loss_no_extra_penalty():
    snap = _base_snapshot()
    events = [
        ArrowHit(owner=0, target=1, part="head", damage=9999.0, x=0, y=0),
        Death(who=1, killer=0, headshot=True, x=0, y=0, facing=1),
        RoundOver(winner=0, p0_rounds=1, p1_rounds=0),
    ]
    reward = compute_reward(events, snap, perspective=1)
    expected = STEP_PENALTY + ROUND_LOSS_PENALTY
    assert reward == pytest.approx(expected)


def test_torso_hit_scales_linearly_with_damage():
    snap = _base_snapshot()
    events = [ArrowHit(owner=0, target=1, part="torso", damage=34.0, x=0, y=0)]
    dealt = compute_reward(events, snap, perspective=0)
    taken = compute_reward(events, snap, perspective=1)
    assert dealt == pytest.approx(STEP_PENALTY + 34.0 * DAMAGE_SCALE)
    assert taken == pytest.approx(STEP_PENALTY - 34.0 * DAMAGE_SCALE)


def test_round_draw_gives_neither_win_nor_loss_reward():
    snap = _base_snapshot()
    events = [RoundOver(winner=None, p0_rounds=0, p1_rounds=0)]
    assert compute_reward(events, snap, 0) == pytest.approx(STEP_PENALTY)
    assert compute_reward(events, snap, 1) == pytest.approx(STEP_PENALTY)


def test_near_miss_bonus_is_zero_when_scale_is_zero():
    snap = _base_snapshot()
    enemy = snap.archers[1]
    on_target_arrow = ArrowState(id=0, x=enemy.x, y=enemy.top() + 18.0, vx=1.0, vy=0.0, owner=0)
    snap_with_arrow = MatchSnapshot(
        arena=snap.arena, archers=snap.archers, arrows=(on_target_arrow,), wind=snap.wind,
        round_time_left=snap.round_time_left, p0_rounds=0, p1_rounds=0,
        match_over=False, winner=None,
    )
    reward = compute_reward([], snap_with_arrow, 0, near_miss_scale=0.0)
    assert reward == pytest.approx(STEP_PENALTY)


def test_near_miss_bonus_grows_as_an_own_arrow_approaches_the_enemy():
    snap = _base_snapshot()
    enemy = snap.archers[1]
    target_x, target_y = enemy.x, enemy.top() + 18.0

    def snapshot_with_arrow_at(dx: float) -> MatchSnapshot:
        arrow = ArrowState(id=0, x=target_x + dx, y=target_y, vx=1.0, vy=0.0, owner=0)
        return MatchSnapshot(
            arena=snap.arena, archers=snap.archers, arrows=(arrow,), wind=snap.wind,
            round_time_left=snap.round_time_left, p0_rounds=0, p1_rounds=0,
            match_over=False, winner=None,
        )
