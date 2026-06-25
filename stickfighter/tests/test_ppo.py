import pytest
import torch

from stickfighter.ai.obs import OBS_DIM
from stickfighter.ai.ppo.policy import ActorCritic
from stickfighter.ai.ppo.ppo import PPOConfig, RolloutBuffer, ppo_update
from stickfighter.core.actions import ACTION_DIM

GAMMA = 0.9
LAMBDA = 0.8


def _buffer_from(rewards, values, dones, obs_dim=1, action_dim=1) -> RolloutBuffer:
    device = torch.device("cpu")
    buf = RolloutBuffer(
        len(rewards), n_envs=1, obs_dim=obs_dim, action_dim=action_dim, device=device
    )
    for r, v, d in zip(rewards, values, dones, strict=True):
        buf.add(
            torch.zeros(1, obs_dim),
            torch.zeros(1, action_dim),
            torch.zeros(1),
            [r],
            [d],
            torch.tensor([v]),
        )
    return buf


def test_gae_matches_hand_computed_values_with_no_episode_boundary():
    # gamma=0.9, lambda=0.8, rewards=[1,2,3], values=[0.5,0.6,0.7], last_value=0.8
    # worked by hand back-to-front: A_2=3.02, A_1=4.2044, A_0=4.067168
    buf = _buffer_from(rewards=[1.0, 2.0, 3.0], values=[0.5, 0.6, 0.7], dones=[0.0, 0.0, 0.0])
    buf.compute_gae(torch.tensor([0.8]), gamma=GAMMA, gae_lambda=LAMBDA)

    expected_advantages = [4.067168, 4.2044, 3.02]
    actual_advantages = buf.advantages.squeeze(-1).tolist()
    for actual, expected in zip(actual_advantages, expected_advantages, strict=True):
        assert actual == pytest.approx(expected, abs=1e-4)
