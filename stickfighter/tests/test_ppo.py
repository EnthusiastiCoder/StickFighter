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
