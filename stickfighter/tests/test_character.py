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

    min_y = archer.y
    landed_step = None
    for i in range(300):
        step_character(archer, NOOP, _FLAT, C.DT)
        min_y = min(min_y, archer.y)
        if archer.grounded:
            landed_step = i
            break

    assert landed_step is not None
    assert archer.y == C.GROUND_Y
    # Semi-implicit Euler at 60Hz systematically undershoots the closed-form
    # continuous apex (v^2/2g) by a few percent -- a discretization artifact,
    # not a bug, so this checks "close to" rather than exact equality.
    expected_apex = C.CHAR_JUMP_SPEED**2 / (2 * C.GRAVITY)
    assert (C.GROUND_Y - min_y) == pytest.approx(expected_apex, rel=0.06)


def test_archer_lands_on_a_floating_platform():
    # twin_mid has a platform at x=200, y=380, width=140 -> spans [200, 340]
    archer = ArcherState(x=270.0, y=100.0)
    _run(archer, _TWIN_MID, NOOP, 200)
    assert archer.grounded
    assert archer.y == 380.0


def test_archer_falls_through_gap_between_platforms_to_ground():
    # x=500 is well outside both twin_mid platforms ([200,340] and [620,760])
    archer = ArcherState(x=500.0, y=100.0)
    _run(archer, _TWIN_MID, NOOP, 400)
    assert archer.grounded
    assert archer.y == C.GROUND_Y


def test_horizontal_movement_is_clamped_to_arena_bounds():
    archer = ArcherState(x=C.CHAR_HALF_WIDTH, y=C.GROUND_Y, grounded=True)
    move_left = Action(move=-1.0, jump=False, aim_angle=0.0, draw=0.0, release=False)
    _run(archer, _FLAT, move_left, 30)
    assert archer.x == C.CHAR_HALF_WIDTH
