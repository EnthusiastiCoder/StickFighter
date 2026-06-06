import math

import numpy as np

from stickfighter.ai.obs import OBS_DIM, encode_obs
from stickfighter.core import constants as C
from stickfighter.core.arena import ARENAS
from stickfighter.core.entities import ArcherState
from stickfighter.core.match import MatchSnapshot

_FLAT = next(a for a in ARENAS if a.name == "flat")


def _symmetric_snapshot(aim_angle: float = 0.4, vx: float = 50.0, rounds: int = 1) -> MatchSnapshot:
    """Two archers that are exact mirror images of each other, on a
    symmetric arena, with no arrows and no wind -- from this state, both
    perspectives should encode to the exact same observation vector.
    """
    a0 = ArcherState(
        x=110.0, y=C.GROUND_Y, vx=vx, vy=0.0, facing=1, grounded=True, hp=70.0,
        reload_timer=0.1, alive=True, aim_angle=aim_angle, draw=0.6, kills=1, coins=3,
    )
    a1 = ArcherState(
        x=850.0, y=C.GROUND_Y, vx=-vx, vy=0.0, facing=-1, grounded=True, hp=70.0,
        reload_timer=0.1, alive=True, aim_angle=math.pi - aim_angle, draw=0.6, kills=1, coins=3,
    )
    return MatchSnapshot(
        arena=_FLAT, archers=(a0, a1), arrows=(), wind=0.0, round_time_left=15.0,
        p0_rounds=rounds, p1_rounds=rounds, match_over=False, winner=None,
    )


def test_mirrored_perspectives_of_a_symmetric_state_produce_identical_obs():
    snap = _symmetric_snapshot()
    o0 = encode_obs(snap, 0)
    o1 = encode_obs(snap, 1)
    assert np.allclose(o0, o1, atol=1e-5)
