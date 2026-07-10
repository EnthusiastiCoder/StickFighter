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
