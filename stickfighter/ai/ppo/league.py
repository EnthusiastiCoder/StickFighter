"""Self-play opponent pool: frozen policy snapshots sampled via PFSP
(prioritized fictitious self-play -- weight opponents the trainee is
currently losing to more heavily), plus Elo ratings to track skill
progression across the league.
"""

import copy
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from stickfighter.ai.obs import encode_obs
from stickfighter.ai.ppo.policy import ActorCritic
from stickfighter.core.actions import Action
from stickfighter.core.match import MatchSnapshot
