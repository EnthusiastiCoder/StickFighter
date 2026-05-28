"""Arrow spawning/integration, swept-segment collision tests, and the inverse
ballistics solver bots use to aim (see solve_launch_angle)."""

import math

from stickfighter.core import constants as C
from stickfighter.core.entities import ArcherState, ArrowState, Circle, Rect


def bow_origin(archer: ArcherState) -> tuple[float, float]:
    x = archer.x + archer.facing * C.BOW_FORWARD_OFFSET
    y = archer.top() + C.BOW_HEIGHT_OFFSET
    return x, y


def spawn_arrow(archer: ArcherState, owner: int, arrow_id: int) -> ArrowState:
    speed = C.ARROW_SPEED_MIN + archer.draw * (C.ARROW_SPEED_MAX - C.ARROW_SPEED_MIN)
    ox, oy = bow_origin(archer)
    vx = math.cos(archer.aim_angle) * speed
    vy = math.sin(archer.aim_angle) * speed
    return ArrowState(id=arrow_id, x=ox, y=oy, vx=vx, vy=vy, owner=owner)
