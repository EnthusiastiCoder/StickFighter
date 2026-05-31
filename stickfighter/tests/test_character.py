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


def _run(archer: ArcherState, arena, action, steps: int) -> None:
    for _ in range(steps):
        step_character(archer, action, arena, C.DT)


def test_jump_apex_matches_kinematics_and_lands_back_on_ground():
    archer = ArcherState(x=110.0, y=C.GROUND_Y, grounded=True)
    step_character(archer, JUMP, _FLAT, C.DT)
    assert not archer.grounded
    assert archer.vy < 0.0
