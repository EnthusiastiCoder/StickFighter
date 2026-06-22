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
