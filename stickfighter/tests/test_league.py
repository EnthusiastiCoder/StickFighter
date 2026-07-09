from stickfighter.ai.obs import OBS_DIM
from stickfighter.ai.ppo.league import DEFAULT_ELO, League
from stickfighter.ai.ppo.policy import ActorCritic
from stickfighter.core.actions import ACTION_DIM
from stickfighter.core.match import MatchSim


def test_empty_league_reports_empty_and_default_elo():
    league = League()
    assert league.is_empty()
    assert league.mean_elo() == DEFAULT_ELO


def test_add_snapshot_seeds_elo_from_league_average():
    league = League()
    model = ActorCritic()
    e1 = league.add_snapshot(model, "a", elo=1400.0)
    e2 = league.add_snapshot(model, "b")  # no explicit elo -> seeded from current average
    assert e1.elo == 1400.0
    assert e2.elo == 1400.0
    assert not league.is_empty()


def test_league_prunes_weakest_entry_beyond_max_size():
    league = League(max_size=2)
    model = ActorCritic()
    league.add_snapshot(model, "weak", elo=1000.0)
    league.add_snapshot(model, "strong", elo=1500.0)
    league.add_snapshot(model, "newest", elo=1200.0)
    assert len(league.entries) == 2
    assert "weak" not in {e.name for e in league.entries}
