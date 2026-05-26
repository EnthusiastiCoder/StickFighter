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


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float


@dataclass(frozen=True)
class Rect:
    left: float
    top: float
    right: float
    bottom: float


def head_hitbox(archer: ArcherState) -> Circle:
    top = archer.top()
    return Circle(cx=archer.x, cy=top + C.HEAD_CENTER_Y, r=C.HEAD_RADIUS)


def torso_hitbox(archer: ArcherState) -> Rect:
    top = archer.top()
    return Rect(
        left=archer.x - C.CHAR_HALF_WIDTH,
        top=top + C.TORSO_TOP_Y,
        right=archer.x + C.CHAR_HALF_WIDTH,
        bottom=top + C.TORSO_BOTTOM_Y,
    )


def limb_hitbox(archer: ArcherState) -> Rect:
    top = archer.top()
    return Rect(
        left=archer.x - C.CHAR_HALF_WIDTH,
        top=top + C.TORSO_BOTTOM_Y,
        right=archer.x + C.CHAR_HALF_WIDTH,
        bottom=top + C.LIMB_BOTTOM_Y,
    )


def body_aabb(archer: ArcherState) -> Rect:
    """Full bounding box, used for character-vs-platform collision."""
    return Rect(
        left=archer.x - C.CHAR_HALF_WIDTH,
        top=archer.top(),
        right=archer.x + C.CHAR_HALF_WIDTH,
        bottom=archer.y,
    )
