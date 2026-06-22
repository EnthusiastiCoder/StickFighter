"""Custom PPO: rollout buffer with GAE, clipped surrogate objective, clipped
value loss, advantage normalization, gradient clipping. Written from scratch
rather than stable-baselines3 so it can drive a self-play league without
fighting someone else's VecEnv/policy abstractions.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from stickfighter.ai.ppo.policy import ActorCritic


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    n_epochs: int = 4
    n_minibatches: int = 4
    normalize_advantages: bool = True
