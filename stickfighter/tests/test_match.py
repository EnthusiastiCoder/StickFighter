import random

from stickfighter.core import constants as C
from stickfighter.core.actions import Action
from stickfighter.core.arena import ARENAS
from stickfighter.core.ballistics import solve_launch_angle
from stickfighter.core.entities import head_hitbox, torso_hitbox
from stickfighter.core.events import ArrowHit, CoinEarned, Death, MatchOver, RoundOver
from stickfighter.core.match import MatchSim

NOOP = Action(move=0.0, jump=False, aim_angle=0.0, draw=0.0, release=False)
_FLAT = next(a for a in ARENAS if a.name == "flat")
