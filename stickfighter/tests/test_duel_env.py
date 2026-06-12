import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from stickfighter.ai.bots.ballistic import HARD, PERFECT, BallisticBot
from stickfighter.ai.bots.idle import IdleBot
from stickfighter.ai.envs.duel_env import DuelEnv
from stickfighter.ai.obs import OBS_DIM
from stickfighter.core.actions import Action
from stickfighter.core.arena import ARENAS
