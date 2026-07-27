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


SWORDSMAN = EnemyConfig(
    name="swordsman", hp=30.0, speed=380.0, melee_range=45.0, melee_damage=25.0, has_shield=False
)
SPEARMAN = EnemyConfig(
    name="spearman", hp=30.0, speed=320.0, melee_range=65.0, melee_damage=30.0, has_shield=False
)
SHIELDBEARER = EnemyConfig(
    name="shieldbearer", hp=30.0, speed=230.0, melee_range=45.0, melee_damage=45.0, has_shield=True
)

SPEED_VARIANCE = (0.8, 1.25)  # per-instance multiplier so a wave doesn't move in lockstep


@dataclass
class MeleeEnemyState:
    enemy_id: int
    config: EnemyConfig
    x: float
    y: float  # feet position, same convention as ArcherState
    vx: float = 0.0
    facing: int = -1  # enemies walk in from the right, facing left, toward the hero
    hp: float = 0.0
    alive: bool = True
    grounded: bool = True  # ground-only entity; kept so draw._draw_legs can be reused as-is
    attack_cooldown: float = 0.0
    speed_multiplier: float = 1.0  # per-instance jitter, see SPEED_VARIANCE

    def speed(self) -> float:
        return self.config.speed * self.speed_multiplier

    def top(self) -> float:
        return self.y - C.CHAR_HEIGHT

    def __post_init__(self) -> None:
        if self.hp <= 0.0:
            self.hp = self.config.hp
