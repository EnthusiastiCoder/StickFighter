from stickfighter.ai.obs import OBS_DIM
from stickfighter.ai.ppo.league import DEFAULT_ELO, League
from stickfighter.ai.ppo.policy import ActorCritic
from stickfighter.core.actions import ACTION_DIM
from stickfighter.core.match import MatchSim


def test_empty_league_reports_empty_and_default_elo():
    league = League()
    assert league.is_empty()
    assert league.mean_elo() == DEFAULT_ELO
