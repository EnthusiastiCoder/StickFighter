"""Actor-critic network: separate policy and value MLPs (2x256, tanh),
state-independent log_std, orthogonal init -- the standard modern
continuous-control PPO recipe (as in CleanRL's ppo_continuous_action.py),
adapted with a final Tanh on the policy mean since our action space is
bounded to [-1, 1] rather than unbounded.
"""

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from stickfighter.ai.obs import OBS_DIM
from stickfighter.core.actions import ACTION_DIM

HIDDEN_SIZE = 256
LOG_STD_INIT = -0.5
_SQRT2 = float(np.sqrt(2.0))


def _orthogonal_layer(in_features: int, out_features: int, gain: float) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            _orthogonal_layer(obs_dim, HIDDEN_SIZE, _SQRT2),
            nn.Tanh(),
            _orthogonal_layer(HIDDEN_SIZE, HIDDEN_SIZE, _SQRT2),
            nn.Tanh(),
            _orthogonal_layer(HIDDEN_SIZE, action_dim, 0.01),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), LOG_STD_INIT))

        self.critic = nn.Sequential(
            _orthogonal_layer(obs_dim, HIDDEN_SIZE, _SQRT2),
            nn.Tanh(),
            _orthogonal_layer(HIDDEN_SIZE, HIDDEN_SIZE, _SQRT2),
            nn.Tanh(),
            _orthogonal_layer(HIDDEN_SIZE, 1, 1.0),
        )
