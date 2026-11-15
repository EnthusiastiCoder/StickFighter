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


def _arena(name: str, floating: list[Platform]) -> Arena:
    p0, p1 = _spawns()
    return Arena(name=name, platforms=(_ground(), *floating), spawn_p0=p0, spawn_p1=p1)


ARENAS: tuple[Arena, ...] = (
    _arena("flat", []),
    _arena("twin_mid", [Platform(200, 380, 140), Platform(620, 380, 140)]),
    _arena(
        "staircase",
        [Platform(150, 420, 120), Platform(690, 420, 120), Platform(410, 300, 140)],
    ),
    _arena("center_tower", [Platform(410, 260, 140)]),
    _arena("low_twins", [Platform(60, 430, 150), Platform(750, 430, 150)]),
    _arena(
        "scattered",
        [
            Platform(120, 400, 110),
            Platform(730, 340, 110),
            Platform(420, 440, 120),
            Platform(300, 260, 100),
        ],
    ),
    _arena(
        "zigzag",
        [
            Platform(80, 400, 130),
            Platform(340, 360, 130),
            Platform(600, 280, 130),
            Platform(750, 400, 130),
        ],
    ),
    _arena("wide_shelf", [Platform(330, 440, 300)]),
)


def random_arena(rng: random.Random) -> Arena:
    return ARENAS[rng.randrange(len(ARENAS))]
