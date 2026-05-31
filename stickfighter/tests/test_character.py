import pytest

from stickfighter.core import constants as C
from stickfighter.core.actions import Action
from stickfighter.core.arena import ARENAS
from stickfighter.core.character import step_character
from stickfighter.core.entities import ArcherState

NOOP = Action(move=0.0, jump=False, aim_angle=0.0, draw=0.0, release=False)
JUMP = Action(move=0.0, jump=True, aim_angle=0.0, draw=0.0, release=False)

_FLAT = next(a for a in ARENAS if a.name == "flat")
_TWIN_MID = next(a for a in ARENAS if a.name == "twin_mid")
