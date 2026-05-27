"""Arrow spawning/integration, swept-segment collision tests, and the inverse
ballistics solver bots use to aim (see solve_launch_angle)."""

import math

from stickfighter.core import constants as C
from stickfighter.core.entities import ArcherState, ArrowState, Circle, Rect


def bow_origin(archer: ArcherState) -> tuple[float, float]:
    x = archer.x + archer.facing * C.BOW_FORWARD_OFFSET
    y = archer.top() + C.BOW_HEIGHT_OFFSET
    return x, y
