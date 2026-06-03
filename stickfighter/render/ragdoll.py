"""Death ragdoll: a short-lived pymunk simulation for corpses.

This is purely cosmetic -- MatchSim has already reset for the next round by
the time a ragdoll plays, so nothing here can feed back into gameplay.
"""

import random

import pygame
import pymunk

from stickfighter.core import constants as C
from stickfighter.core.events import Death, Event
from stickfighter.render import theme

RAGDOLL_LIFETIME = 2.5
LINE_WIDTH = 5
_GROUP = 1  # ragdoll pieces never collide with each other or other ragdolls


class Ragdoll:
    def __init__(self, x: float, y: float, facing: int, knock_vx: float, knock_vy: float) -> None:
        self.age = 0.0
        self._rng = random.Random()
        self.space = pymunk.Space()
        self.space.gravity = (0.0, C.GRAVITY)

        ground_pts = ((-2000.0, C.GROUND_Y), (4000.0, C.GROUND_Y))
        ground = pymunk.Segment(self.space.static_body, *ground_pts, 4.0)
        ground.friction = 0.9
        self.space.add(ground)
