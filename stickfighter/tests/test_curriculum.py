from stickfighter.ai.ppo.curriculum import SELFPLAY_STAGE_NAME, STAGES, Curriculum


def test_starts_at_the_first_stage():
    c = Curriculum()
    assert c.stage.name == STAGES[0].name
    assert c.stage_idx == 0
