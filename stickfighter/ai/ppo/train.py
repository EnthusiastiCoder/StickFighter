"""PPO training CLI: curriculum-driven learning from scripted bots through a
self-play league of frozen past checkpoints.

Uses gymnasium's SyncVectorEnv exclusively. AsyncVectorEnv (real
multiprocessing workers, see envs/vec.py) was benchmarked against it and did
NOT outperform it at this project's scale -- the sim + bot computation is
cheap enough that multiprocessing IPC overhead ate the theoretical
parallelism gain. Running everything in one process also means a self-play
opponent's weights can just be swapped in place (`sub_env.opponent = ...`),
with no cross-process weight-sync mechanism needed at all.
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from stickfighter.ai.bots.base import Policy
from stickfighter.ai.bots.idle import IdleBot
from stickfighter.ai.envs.duel_env import DuelEnv
from stickfighter.ai.obs import OBS_DIM
from stickfighter.ai.ppo.curriculum import SELFPLAY_STAGE_NAME, Curriculum
from stickfighter.ai.ppo.league import DEFAULT_ELO, League
from stickfighter.ai.ppo.logging import TrainingLogger
from stickfighter.ai.ppo.policy import ActorCritic, RunningNormalizer
from stickfighter.ai.ppo.ppo import PPOConfig, RolloutBuffer, ppo_update
from stickfighter.ai.seeding import set_seed
from stickfighter.core.actions import ACTION_DIM
