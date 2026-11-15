import torch

from stickfighter.ai.bots.idle import IdleBot
from stickfighter.ai.envs.duel_env import DuelEnv
from stickfighter.ai.obs import OBS_DIM
from stickfighter.ai.ppo.curriculum import STAGES, Curriculum
from stickfighter.ai.ppo.league import League
from stickfighter.ai.ppo.policy import ActorCritic
from stickfighter.ai.ppo.train import (
    build_opponent,
    find_latest_checkpoint,
    linear_schedule,
    make_run_dir,
    save_checkpoint,
)
from stickfighter.core.actions import ACTION_DIM


def test_linear_schedule_interpolates_and_clamps():
    assert linear_schedule(1.0, 0.0, 0.0) == 1.0
    assert linear_schedule(1.0, 0.0, 1.0) == 0.0
    assert linear_schedule(1.0, 0.0, 0.5) == 0.5
    assert linear_schedule(1.0, 0.0, -1.0) == 1.0  # clamped
    assert linear_schedule(1.0, 0.0, 2.0) == 0.0  # clamped


def test_find_latest_checkpoint_is_numeric_not_lexicographic(tmp_path):
    (tmp_path / "ckpt_000000000009.pt").write_bytes(b"")
    (tmp_path / "ckpt_000000000100.pt").write_bytes(b"")
    (tmp_path / "ckpt_000000000050.pt").write_bytes(b"")
    latest = find_latest_checkpoint(tmp_path)
    assert latest.name == "ckpt_000000000100.pt"


def test_find_latest_checkpoint_returns_none_when_empty(tmp_path):
    assert find_latest_checkpoint(tmp_path) is None


def test_build_opponent_uses_scripted_bot_outside_selfplay():
    curriculum = Curriculum()  # starts at S0_static
    league = League()
    model = ActorCritic()

    opponent, entry, near_miss_scale = build_opponent(curriculum, league, model, trainee_elo=1200.0)
    assert entry is None
    assert opponent is not None
    assert near_miss_scale == curriculum.stage.near_miss_scale


def test_build_opponent_seeds_and_samples_league_in_selfplay():
    curriculum = Curriculum(stage_idx=len(STAGES) - 1)
    league = League()
    model = ActorCritic()
    assert league.is_empty()

    opponent, entry, near_miss_scale = build_opponent(curriculum, league, model, trainee_elo=1200.0)
    assert not league.is_empty()  # the trainee itself was auto-seeded as the first entry
    assert entry is not None
    assert near_miss_scale == curriculum.stage.near_miss_scale


def test_build_opponent_matches_near_miss_scale_to_the_rehearsed_stage_not_the_current_one():
    # S2's near_miss_scale is 0.0; a rehearsal match against S0's opponent
    # (IdleBot) must still carry S0's near_miss_scale (1.0), or the whole
    # point of rehearsing an easier stage -- getting shaping signal back --
    # is lost.
    curriculum = Curriculum(stage_idx=2)
    assert curriculum.stage.near_miss_scale == 0.0
    league = League()
    model = ActorCritic()

    seen_scales = set()
    for _ in range(200):
        _opponent, _entry, near_miss_scale = build_opponent(
            curriculum, league, model, trainee_elo=1200.0
        )
        seen_scales.add(near_miss_scale)
    assert seen_scales == {s.near_miss_scale for s in STAGES[: curriculum.stage_idx + 1]}


def test_checkpoint_round_trip_reproduces_identical_actions(tmp_path):
    torch.manual_seed(0)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    curriculum = Curriculum()

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model, optimizer, 1234, curriculum, trainee_elo=1250.0)

    reloaded = ActorCritic()
    ckpt = torch.load(ckpt_path, weights_only=False)
    reloaded.load_state_dict(ckpt["model"])
    assert ckpt["global_step"] == 1234
    assert ckpt["trainee_elo"] == 1250.0
    assert ckpt["stage_idx"] == curriculum.stage_idx

    obs = torch.randn(4, OBS_DIM)
    with torch.no_grad():
        _, _, _, value_before = model.get_action_and_value(obs)
        _, _, _, value_after = reloaded.get_action_and_value(obs)
    assert torch.allclose(value_before, value_after)
    assert torch.allclose(model.actor(obs), reloaded.actor(obs))


def test_duel_env_construction_matches_train_loop_shapes():
    # sanity: the exact construction pattern train.py uses for its sub-envs
    env = DuelEnv(opponent=IdleBot())
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert env.action_space.shape == (ACTION_DIM,)


def test_make_run_dir_reuses_the_resumed_directory_by_default(tmp_path):
    # a bare --resume with no --run-dir must keep writing into the resumed
    # run, not silently fork a fresh runs/<timestamp>/ that leaves the
    # checkpoints difficulty.py looks for stale forever.
    resumed_from = tmp_path / "stickfighter_main"
    run_dir = make_run_dir(None, str(resumed_from))
    assert run_dir == resumed_from


def test_make_run_dir_explicit_run_dir_wins_over_resume(tmp_path):
    resumed_from = tmp_path / "stickfighter_main"
    explicit = tmp_path / "explicit_dir"
    run_dir = make_run_dir(str(explicit), str(resumed_from))
    assert run_dir == explicit


def test_make_run_dir_generates_a_timestamped_dir_with_neither_arg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = make_run_dir(None, None)
    assert run_dir.parent.name == "runs"
    assert run_dir.name.startswith("stickfighter_")
