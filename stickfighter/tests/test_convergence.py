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

    device = torch.device("cpu")
    model = ActorCritic(OBS_DIM, ACTION_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    normalizer = RunningNormalizer((OBS_DIM,))
    config = PPOConfig()
    buffer = RolloutBuffer(n_steps, n_envs, OBS_DIM, ACTION_DIM, device)

    global_step = 0
    rollout_mean_rewards = []
    final_entropy = None
    while global_step < total_steps:
        buffer.reset()
        rollout_rewards = []
        for _ in range(n_steps):
            normalizer.update(obs)
            obs_t = torch.as_tensor(normalizer.normalize(obs), dtype=torch.float32, device=device)
            with torch.no_grad():
                action, log_prob, _entropy, value = model.get_action_and_value(obs_t)
            next_obs, reward, terminated, truncated, _infos = vec_env.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            buffer.add(obs_t, action, log_prob, reward, done.astype(np.float32), value)
            rollout_rewards.append(reward.mean())
            global_step += n_envs
            obs = next_obs

        with torch.no_grad():
            last_obs_t = torch.as_tensor(normalizer.normalize(obs), dtype=torch.float32)
            last_value = model.get_value(last_obs_t)
        buffer.compute_gae(last_value, config.gamma, config.gae_lambda)
        stats = ppo_update(model, optimizer, buffer, config)
        final_entropy = stats["entropy"]
        rollout_mean_rewards.append(float(np.mean(rollout_rewards)))

    early = np.mean(rollout_mean_rewards[:3])
    late = np.mean(rollout_mean_rewards[-3:])

    # a working policy on this task (headshot-instakill on a stationary
    # target, with near-miss shaping) reliably improves by >0.001 in mean
    # per-step reward within 100k steps -- observed in practice ~0.0035.
    assert late - early > 0.001
    # entropy on a 5-dim Gaussian starts near ln(2*pi*e*std^2)*5/2 ~= 4.6;
    # collapsing well below 2.0 this early would mean the policy went
    # deterministic almost immediately, a classic PPO failure mode.
    assert final_entropy > 2.0
