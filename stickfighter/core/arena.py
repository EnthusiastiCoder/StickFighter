"""Arena layouts: the ground plus a handful of floating platforms.

Platforms are one-way: an archer only lands on the top surface while falling
onto it (see character.py); there is no collision from the sides or below.
"""

import random
from dataclasses import dataclass

from stickfighter.core import constants as C


@dataclass(frozen=True)
class Platform:
    x: float  # left edge
    y: float  # top (walkable) surface
    width: float

    @property
    def right(self) -> float:
        return self.x + self.width


@dataclass(frozen=True)
class Arena:
    name: str
    platforms: tuple[Platform, ...]  # includes the ground
    spawn_p0: tuple[float, float]
    spawn_p1: tuple[float, float]


def _ground() -> Platform:
    return Platform(x=0.0, y=C.GROUND_Y, width=C.ARENA_WIDTH)


def _spawns(margin: float = 110.0) -> tuple[tuple[float, float], tuple[float, float]]:
    return (margin, C.GROUND_Y), (C.ARENA_WIDTH - margin, C.GROUND_Y)
