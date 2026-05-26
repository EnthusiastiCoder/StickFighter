"""Plain-data state for archers and arrows, plus hitbox geometry queries."""

from dataclasses import dataclass

from stickfighter.core import constants as C


@dataclass
class ArcherState:
    x: float
    y: float  # feet position (bottom of bounding box)
    vx: float = 0.0
    vy: float = 0.0
    facing: int = 1  # +1 right, -1 left
    grounded: bool = False
    hp: float = C.CHAR_MAX_HP
    reload_timer: float = 0.0
    alive: bool = True
    aim_angle: float = 0.0
    draw: float = 0.0
    kills: int = 0
    coins: int = 0
    rounds_won: int = 0

    def top(self) -> float:
        return self.y - C.CHAR_HEIGHT


@dataclass
class ArrowState:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    owner: int
    age: float = 0.0
    alive: bool = True
