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
