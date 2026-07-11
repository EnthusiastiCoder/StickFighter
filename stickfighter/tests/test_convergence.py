"""Slow integration test: PPO must show real, stable learning progress on
S0 (static target practice) within a bounded step budget. Not a "solves
reliably" bar (that needs far more than 100k steps for a 5D continuous
action space) -- a directly measurable floor that would fail if the
training loop were silently broken (wrong GAE, a reward sign flip, an
entropy collapse, exploding KL, etc).

Excluded from the default test run (pyproject.toml's pytest markers select
"not slow" by default) since it takes on the order of a minute.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

from stickfighter.ai.bots.idle import IdleBot
from stickfighter.ai.envs.duel_env import DuelEnv
from stickfighter.ai.obs import OBS_DIM
from stickfighter.ai.ppo.policy import ActorCritic, RunningNormalizer
from stickfighter.ai.ppo.ppo import PPOConfig, RolloutBuffer, ppo_update
from stickfighter.core.actions import ACTION_DIM


@pytest.mark.slow
def test_ppo_shows_measurable_learning_progress_on_s0_within_100k_steps():
    torch.manual_seed(0)
    np.random.seed(0)

    n_envs, n_steps, total_steps = 8, 128, 100_000
    envs = [DuelEnv(opponent=IdleBot(), near_miss_scale=1.0) for _ in range(n_envs)]
    vec_env = gym.vector.SyncVectorEnv([(lambda e=e: e) for e in envs])
    obs, _ = vec_env.reset(seed=0)
