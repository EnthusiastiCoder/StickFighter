"""Kinematic archer controller: horizontal movement, gravity, jumping, and
one-way platform landing. Combat (reload/fire/damage) lives in match.py."""

import math

from stickfighter.core import constants as C
from stickfighter.core.arena import Arena
from stickfighter.core.entities import ArcherState
