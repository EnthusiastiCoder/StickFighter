"""The renderer contract: reading a MatchSnapshot to draw a frame must never
mutate the sim state underneath it (render/ only ever reads)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

from stickfighter.core import constants as C  # noqa: E402
from stickfighter.core.match import MatchSim  # noqa: E402
from stickfighter.render import draw, effects, hud, ragdoll  # noqa: E402

pygame.init()


def _random_played_sim(seed: int, steps: int = 90) -> MatchSim:
    import random

    from stickfighter.core.actions import Action
