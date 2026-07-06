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
