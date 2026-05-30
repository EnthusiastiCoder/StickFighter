import math

import pytest

from stickfighter.core import constants as C
from stickfighter.core.ballistics import (
    segment_hits_circle,
    segment_hits_rect,
    solve_launch_angle,
)
from stickfighter.core.entities import Circle, Rect
