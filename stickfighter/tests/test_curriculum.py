import random

from stickfighter.ai.ppo.curriculum import SELFPLAY_STAGE_NAME, STAGES, Curriculum


def test_starts_at_the_first_stage():
    c = Curriculum()
    assert c.stage.name == STAGES[0].name
    assert c.stage_idx == 0


def test_does_not_advance_before_min_matches_reached():
    c = Curriculum(window=10)
    for _ in range(10):
        advanced = c.record_match(True)
    assert not advanced
    assert c.stage_idx == 0  # S0's min_matches is 50


def test_advances_once_win_rate_and_min_matches_are_both_satisfied():
    c = Curriculum(window=10)
    advanced_flags = [c.record_match(True) for _ in range(50)]
    assert any(advanced_flags)
    assert c.stage_idx == 1


def test_does_not_advance_with_a_low_win_rate_even_past_min_matches():
    c = Curriculum(window=10)
    for _ in range(200):
        c.record_match(False)
    assert c.stage_idx == 0
    assert c.win_rate == 0.0


def test_win_rate_is_a_rolling_window_not_a_lifetime_average():
    c = Curriculum(window=5)
    for _ in range(20):
        c.record_match(False)
    for _ in range(5):
        c.record_match(True)
    assert c.win_rate == 1.0  # only the last 5 (the window) count


def test_last_stage_is_selfplay_and_never_auto_advances_further():
    c = Curriculum(stage_idx=len(STAGES) - 1)
    assert c.stage.name == SELFPLAY_STAGE_NAME
    assert c.stage.opponent_factory is None
    for _ in range(1000):
        advanced = c.record_match(True)
    assert not advanced
    assert c.stage_idx == len(STAGES) - 1


def test_stage_advance_resets_the_rolling_window():
    c = Curriculum(window=10)
    for _ in range(50):
        c.record_match(True)
    assert c.stage_idx == 1
    assert c.results == []
    assert c.matches_this_stage == 0


def test_opponent_stage_is_always_current_stage_at_s0():
    c = Curriculum(stage_idx=0)
    rng = random.Random(0)
    for _ in range(100):
        assert c.opponent_stage(rng).name == STAGES[0].name  # nothing earlier to rehearse


def test_opponent_stage_sometimes_rehearses_an_earlier_stage():
    c = Curriculum(stage_idx=2)
    rng = random.Random(0)
    seen = {c.opponent_stage(rng).name for _ in range(200)}
    assert seen == {STAGES[0].name, STAGES[1].name, STAGES[2].name}


def test_opponent_stage_mostly_returns_the_current_stage():
    c = Curriculum(stage_idx=2)
    rng = random.Random(0)
    current = sum(1 for _ in range(500) if c.opponent_stage(rng).name == STAGES[2].name)
    assert current > 300  # roughly 1 - REHEARSAL_PROB of 500 trials
