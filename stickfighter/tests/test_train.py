import torch

from stickfighter.ai.bots.idle import IdleBot
from stickfighter.ai.envs.duel_env import DuelEnv
from stickfighter.ai.obs import OBS_DIM
from stickfighter.ai.ppo.curriculum import STAGES, Curriculum
from stickfighter.ai.ppo.league import League
from stickfighter.ai.ppo.policy import ActorCritic, RunningNormalizer
from stickfighter.ai.ppo.train import (
    build_opponent,
    find_latest_checkpoint,
    linear_schedule,
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
    normalizer = RunningNormalizer((OBS_DIM,))

    opponent, entry = build_opponent(curriculum, league, model, normalizer, trainee_elo=1200.0)
    assert entry is None
    assert opponent is not None


def test_build_opponent_seeds_and_samples_league_in_selfplay():
    curriculum = Curriculum(stage_idx=len(STAGES) - 1)
    league = League()
    model = ActorCritic()
    normalizer = RunningNormalizer((OBS_DIM,))
    assert league.is_empty()

    opponent, entry = build_opponent(curriculum, league, model, normalizer, trainee_elo=1200.0)
    assert not league.is_empty()  # the trainee itself was auto-seeded as the first entry
    assert entry is not None
    assert opponent is not None


def test_checkpoint_round_trip_reproduces_identical_actions(tmp_path):
    torch.manual_seed(0)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    normalizer = RunningNormalizer((OBS_DIM,))
    curriculum = Curriculum()

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model, optimizer, 1234, normalizer, curriculum, trainee_elo=1250.0)
