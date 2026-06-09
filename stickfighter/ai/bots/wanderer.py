"""Moves and fires randomly, with no aiming skill at all. A very easy sanity
opponent -- one step up from IdleBot, useful for verifying an env/bot wiring
actually produces damage without needing a real ballistics solver."""

import math
import random

from stickfighter.core import constants as C
from stickfighter.core.actions import Action
from stickfighter.core.match import MatchSnapshot


class WandererBot:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._move = 0.0
        self._change_timer = 0.0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)
        self._move = 0.0
        self._change_timer = 0.0

    def act(self, snapshot: MatchSnapshot, perspective: int) -> Action:
        self._change_timer -= C.DT
        if self._change_timer <= 0.0:
            self._move = self._rng.choice((-1.0, 0.0, 1.0))
            self._change_timer = self._rng.uniform(0.4, 1.2)

        jump = self._rng.random() < 0.01
        aim_angle = self._rng.uniform(-math.pi, math.pi)
        draw = self._rng.random()
        release = self._rng.random() < 0.03
        return Action(move=self._move, jump=jump, aim_angle=aim_angle, draw=draw, release=release)
