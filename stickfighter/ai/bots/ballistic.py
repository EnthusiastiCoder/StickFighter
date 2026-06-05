"""Scripted archer that aims with the closed-form ballistics solver. Serves
as both a training-curriculum opponent and a human difficulty ladder --
difficulty is entirely a matter of the noise/delay/target knobs below, not a
different algorithm.
"""

import math
import random
from dataclasses import dataclass

from stickfighter.core import constants as C
from stickfighter.core.actions import Action
from stickfighter.core.ballistics import solve_launch_angle
from stickfighter.core.entities import ArcherState, head_hitbox, torso_hitbox
from stickfighter.core.match import MatchSnapshot
