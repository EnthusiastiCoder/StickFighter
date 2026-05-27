"""Kinematic archer controller: horizontal movement, gravity, jumping, and
one-way platform landing. Combat (reload/fire/damage) lives in match.py."""

import math

from stickfighter.core import constants as C
from stickfighter.core.arena import Arena
from stickfighter.core.entities import ArcherState


def step_character(archer: ArcherState, action, arena: Arena, dt: float) -> None:
    """Mutates archer in place. A dead archer is frozen -- the round-reset in
    match.py is what revives it, and the death ragdoll is a render-only effect."""
    if not archer.alive:
        return

    archer.aim_angle = action.aim_angle
    archer.draw = action.draw
    archer.facing = 1 if math.cos(action.aim_angle) >= 0.0 else -1

    archer.vx = action.move * C.CHAR_MOVE_SPEED
    if action.jump and archer.grounded:
        archer.vy = -C.CHAR_JUMP_SPEED
        archer.grounded = False

    archer.vy = min(archer.vy + C.GRAVITY * dt, C.CHAR_MAX_FALL_SPEED)
