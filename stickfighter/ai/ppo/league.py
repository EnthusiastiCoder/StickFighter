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

DEFAULT_ELO = 1200.0
ELO_K = 24.0


@dataclass
class LeagueEntry:
    name: str
    state_dict: dict
    elo: float = DEFAULT_ELO
    games: int = 0
    wins_vs_trainee: int = 0


@dataclass
class FrozenPolicyBot:
    """Wraps a frozen ActorCritic snapshot as a Policy for match play."""

    model: ActorCritic
    obs_normalizer: object | None = field(default=None)

    def __post_init__(self) -> None:
        self.model.eval()

    def reset(self, seed: int | None = None) -> None:
        pass
