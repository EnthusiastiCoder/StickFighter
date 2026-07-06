"""Evaluate a trained checkpoint: win rates against the scripted bot ladder,
and an optional recorded match GIF/MP4 using the real pygame renderer -- the
"cheap sim trains, pretty renderer records" split, driven by the exact same
MatchSim used everywhere else in this project.
"""

import argparse

import torch

from stickfighter.ai.bots.ballistic import EASY, HARD, MEDIUM, PERFECT, BallisticBot
from stickfighter.ai.bots.idle import IdleBot
from stickfighter.ai.obs import OBS_DIM, encode_obs
from stickfighter.ai.ppo.policy import ActorCritic, RunningNormalizer
from stickfighter.core.actions import ACTION_DIM, Action
from stickfighter.core.match import MatchSim

BOT_LADDER = {
    "idle": IdleBot,
    "easy": lambda: BallisticBot(EASY),
    "medium": lambda: BallisticBot(MEDIUM),
    "hard": lambda: BallisticBot(HARD),
    "perfect": lambda: BallisticBot(PERFECT),
}
