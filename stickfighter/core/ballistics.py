"""Arrow spawning/integration, swept-segment collision tests, and the inverse
ballistics solver bots use to aim (see solve_launch_angle)."""

import math

from stickfighter.core import constants as C
from stickfighter.core.entities import ArcherState, ArrowState, Circle, Rect
