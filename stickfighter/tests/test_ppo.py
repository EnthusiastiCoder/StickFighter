import pytest
import torch

from stickfighter.ai.obs import OBS_DIM
from stickfighter.ai.ppo.policy import ActorCritic
from stickfighter.ai.ppo.ppo import PPOConfig, RolloutBuffer, ppo_update
from stickfighter.core.actions import ACTION_DIM

GAMMA = 0.9
LAMBDA = 0.8
