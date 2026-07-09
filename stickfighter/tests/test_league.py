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


def test_frozen_policy_bot_produces_a_valid_action():
    league = League()
    model = ActorCritic()
    entry = league.add_snapshot(model, "ckpt")
    bot = league.make_opponent(entry, OBS_DIM, ACTION_DIM)

    sim = MatchSim()
    sim.reset(seed=0)
    action = bot.act(sim.snapshot(), 1)
    assert action is not None
    assert -1.0 <= action.aim_angle / 3.14159265 <= 1.0


def test_record_match_updates_win_tracking_and_elo_symmetrically():
    league = League()
    model = ActorCritic()
    entry = league.add_snapshot(model, "ckpt", elo=1200.0)
    trainee_elo = 1200.0

    new_trainee_elo = league.record_match(entry, trainee_elo, entry_won=True)
    assert entry.games == 1
    assert entry.wins_vs_trainee == 1
    assert entry.elo > 1200.0  # entry gained rating for beating an equally-rated trainee
    assert new_trainee_elo < 1200.0  # trainee lost rating symmetrically


def test_record_match_trainee_win_lowers_entry_elo():
    league = League()
    model = ActorCritic()
    entry = league.add_snapshot(model, "ckpt", elo=1200.0)

    new_trainee_elo = league.record_match(entry, 1200.0, entry_won=False)
    assert entry.wins_vs_trainee == 0
    assert entry.games == 1
    assert entry.elo < 1200.0
    assert new_trainee_elo > 1200.0


def test_pfsp_sampling_favors_entries_the_trainee_loses_to():
    league = League(pfsp_power=4.0, seed=0)
    model = ActorCritic()
    strong = league.add_snapshot(model, "beats_trainee_always")
    weak = league.add_snapshot(model, "never_beats_trainee")
