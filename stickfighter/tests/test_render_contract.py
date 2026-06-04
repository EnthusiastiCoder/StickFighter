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

    rng = random.Random(seed)
    sim = MatchSim()
    sim.reset(seed=seed)
    for _ in range(steps):
        if sim.match_over:
            break
        a0 = Action(
            move=rng.uniform(-1, 1),
            jump=rng.random() < 0.05,
            aim_angle=rng.uniform(-3.14, 3.14),
            draw=rng.random(),
            release=rng.random() < 0.1,
        )
        a1 = Action(
            move=rng.uniform(-1, 1),
            jump=rng.random() < 0.05,
            aim_angle=rng.uniform(-3.14, 3.14),
            draw=rng.random(),
            release=rng.random() < 0.1,
        )
        sim.step(a0, a1)
    return sim
