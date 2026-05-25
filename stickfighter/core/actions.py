"""The single action primitive shared by human input and the AI's policy output.

A human plays via drag-to-aim: dragging from the archer sets aim_angle/draw,
releasing fires. The AI's continuous Box(5) action mirrors this exactly, so
the trained policy plays the same control surface a human does.
"""

import math
from dataclasses import dataclass

import numpy as np

ACTION_DIM = 5


@dataclass
class Action:
    move: float  # [-1, 1], horizontal move intent
    jump: bool
    aim_angle: float  # radians, [-pi, pi], y-down convention (positive = downward)
    draw: float  # [0, 1], bow draw fraction
    release: bool

    @staticmethod
    def from_array(a: np.ndarray) -> "Action":
        """Decode a raw policy/env action: every dim lives in [-1, 1] (a
        single Gaussian head is simpler and more stable than a mixed-range
        one), so draw's logical [0, 1] range is the affine remap below."""
        move = float(np.clip(a[0], -1.0, 1.0))
        jump = bool(a[1] > 0.0)
        aim_angle = float(np.clip(a[2], -1.0, 1.0)) * math.pi
        draw = float(np.clip((a[3] + 1.0) / 2.0, 0.0, 1.0))
        release = bool(a[4] > 0.0)
        return Action(move=move, jump=jump, aim_angle=aim_angle, draw=draw, release=release)

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.move,
                1.0 if self.jump else -1.0,
                self.aim_angle / math.pi,
                self.draw * 2.0 - 1.0,
                1.0 if self.release else -1.0,
            ],
            dtype=np.float32,
        )


NOOP = Action(move=0.0, jump=False, aim_angle=0.0, draw=0.0, release=False)
