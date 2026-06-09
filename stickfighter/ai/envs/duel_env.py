"""Gymnasium environment for 1v1 duels: the learning agent controls one
archer against a configurable opponent Policy (a scripted bot during
curriculum stages, another network snapshot during self-play).

Frame-skip repeats the agent's action for `frame_skip` sim steps (60Hz sim ->
30Hz decisions by default), which is plenty of control resolution for this
game and roughly doubles throughput. The opponent is re-queried every sim
step regardless of frame_skip, so it stays fully reactive.
"""

from typing import Any

import gymnasium as gym
import numpy as np

from stickfighter.ai.bots.base import Policy
from stickfighter.ai.bots.idle import IdleBot
from stickfighter.ai.obs import OBS_DIM, encode_obs
from stickfighter.ai.reward import compute_reward
from stickfighter.core.actions import Action
from stickfighter.core.arena import Arena
from stickfighter.core.events import Event
from stickfighter.core.match import MatchSim

DEFAULT_FRAME_SKIP = 2
DEFAULT_MAX_EPISODE_STEPS = 3600  # 60s of sim time at 60Hz
