"""Encodes a MatchSnapshot into a compact, egocentric observation vector.

The observation is always from the acting archer's point of view and always
mirrored so archer 1 sees a canonical "I am on the left, enemy is to my
right" frame -- the same policy can then play either side, doubling the
effective self-play data per match.
"""

import math

import numpy as np

from stickfighter.core import constants as C
from stickfighter.core.match import MatchSnapshot

MAX_ARROWS = C.MAX_ARROWS_TRACKED
MAX_PLATFORMS = 4
CHAR_VEL_NORM = C.CHAR_JUMP_SPEED
ARROW_VEL_NORM = C.ARROW_SPEED_MAX
