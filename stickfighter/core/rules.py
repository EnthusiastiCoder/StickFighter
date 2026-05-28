"""Pure damage/scoring/win-condition lookups -- no mutable state."""

from stickfighter.core import constants as C

_DAMAGE = {"head": C.DAMAGE_HEAD, "torso": C.DAMAGE_TORSO, "limb": C.DAMAGE_LIMB}
_COINS = {"head": C.COINS_HEADSHOT, "torso": C.COINS_TORSO, "limb": C.COINS_LIMB}


def damage_for_part(part: str) -> float:
    return _DAMAGE[part]


def coins_for_part(part: str) -> int:
    return _COINS[part]


def is_match_won(p0_rounds: int, p1_rounds: int) -> int | None:
    if p0_rounds >= C.WIN_ROUNDS:
        return 0
    if p1_rounds >= C.WIN_ROUNDS:
        return 1
    return None
