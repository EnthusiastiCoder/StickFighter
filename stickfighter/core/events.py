"""Events emitted by MatchSim.step(). Consumed by the renderer, reward function,
and logging -- never by the sim itself, so they can be frozen/immutable."""

from dataclasses import dataclass

Part = str  # one of "head", "torso", "limb"


@dataclass(frozen=True)
class ArrowFired:
    owner: int
    x: float
    y: float
    vx: float
    vy: float


@dataclass(frozen=True)
class ArrowHit:
    owner: int  # archer who fired the arrow
    target: int  # archer who was hit
    part: Part
    damage: float
    x: float
    y: float


@dataclass(frozen=True)
class ArrowMiss:
    owner: int
    x: float
    y: float


@dataclass(frozen=True)
class Death:
    who: int
    killer: int
    headshot: bool
    x: float  # position at the moment of death -- the round resets right
    y: float  # after this event, so the renderer has nowhere else to get
    facing: int  # the corpse's spawn pose for its death ragdoll


@dataclass(frozen=True)
class CoinEarned:
    who: int
    amount: int


@dataclass(frozen=True)
class RoundOver:
    winner: int | None  # None on a timeout draw
    p0_rounds: int
    p1_rounds: int


@dataclass(frozen=True)
class MatchOver:
    winner: int


Event = ArrowFired | ArrowHit | ArrowMiss | Death | CoinEarned | RoundOver | MatchOver
