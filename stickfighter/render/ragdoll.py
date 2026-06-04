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

        top = y - C.CHAR_HEIGHT
        head_pt = (x, top + C.HEAD_CENTER_Y)
        shoulder_pt = (x, top + C.TORSO_TOP_Y)
        hip_pt = (x, top + C.TORSO_BOTTOM_Y)
        foot1_pt = (x + 9.0 * facing, y)
        foot2_pt = (x - 7.0 * facing, y)

        self.head_body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0.0, C.HEAD_RADIUS))
        self.head_body.position = head_pt
        head_shape = pymunk.Circle(self.head_body, C.HEAD_RADIUS)
        head_shape.friction = 0.5
        head_shape.filter = pymunk.ShapeFilter(group=_GROUP)
        self.space.add(self.head_body, head_shape)

        self.torso_body, self._torso_local = self._add_segment(shoulder_pt, hip_pt, 3.0)
        self.leg1_body, self._leg1_local = self._add_segment(hip_pt, foot1_pt, 1.4)
        self.leg2_body, self._leg2_local = self._add_segment(hip_pt, foot2_pt, 1.4)

        self.space.add(pymunk.PivotJoint(self.head_body, self.torso_body, shoulder_pt))
        self.space.add(pymunk.PivotJoint(self.torso_body, self.leg1_body, hip_pt))
        self.space.add(pymunk.PivotJoint(self.torso_body, self.leg2_body, hip_pt))

        for body in (self.head_body, self.torso_body, self.leg1_body, self.leg2_body):
            body.velocity = (knock_vx * 0.25, knock_vy * 0.25 - 60.0)
            body.angular_velocity = self._rng.uniform(-5.0, 5.0)
