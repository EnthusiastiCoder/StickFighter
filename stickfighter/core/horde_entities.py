"""Plain-data state for melee horde enemies, plus hardcoded type presets.

Mirrors the spirit of bots/ballistic.py's difficulty presets, but for melee
threat instead of archer accuracy: each preset is just a handful of numbers
(hp, speed, reach, damage, shield), no separate AI per type. Deliberately a
separate dataclass from ArcherState (not a reuse/subclass) -- these enemies
have no bow, no draw/reload cycle, and a shield flag that has no meaning for
an archer, so folding them into ArcherState would just add dead fields there.
"""

from dataclasses import dataclass

from stickfighter.core import constants as C


@dataclass(frozen=True)
class EnemyConfig:
    name: str
    hp: float
    speed: float  # px/s, walking toward the hero
    melee_range: float  # px from the hero at which it stops and attacks
    melee_damage: float
    has_shield: bool  # blocks torso/limb hits; a headshot still kills
