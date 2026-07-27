"""Plain-data state for melee horde enemies, plus hardcoded type presets.

Mirrors the spirit of bots/ballistic.py's difficulty presets, but for melee
threat instead of archer accuracy: each preset is just a handful of numbers
(hp, speed, reach, damage, shield), no separate AI per type. Deliberately a
separate dataclass from ArcherState (not a reuse/subclass) -- these enemies
have no bow, no draw/reload cycle, and a shield flag that has no meaning for
an archer, so folding them into ArcherState would just add dead fields there.
"""

from dataclasses import dataclass, field

from stickfighter.core import constants as C
from stickfighter.core.entities import ArcherState


@dataclass(frozen=True)
class EnemyConfig:
    name: str
    hp: float
    speed: float  # px/s, walking toward the hero
    melee_range: float  # px from the hero at which it stops and attacks
    melee_damage: float
    has_shield: bool  # blocks torso/limb hits; a headshot still kills


SWORDSMAN = EnemyConfig(
    name="swordsman", hp=30.0, speed=320.0, melee_range=45.0, melee_damage=25.0, has_shield=False
)
SPEARMAN = EnemyConfig(
    name="spearman", hp=30.0, speed=270.0, melee_range=65.0, melee_damage=30.0, has_shield=False
)
SHIELDBEARER = EnemyConfig(
    name="shieldbearer", hp=30.0, speed=195.0, melee_range=45.0, melee_damage=45.0, has_shield=True
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


CLOUD_ARCHER_HP = 30.0
CLOUD_ARCHER_HEIGHT = 165.0  # px above the ground the cloud (and archer) hovers
CLOUD_ARCHER_STANDOFF = (420.0, 620.0)  # stops drifting once this close to the hero
CLOUD_ARCHER_DRIFT_SPEED = 150.0
CLOUD_ARCHER_FIRE_COOLDOWN = (1.3, 1.9)  # randomized per shot, not a robotic metronome
CLOUD_ARCHER_AIM_NOISE = 0.10  # radians, gaussian -- real but beatable accuracy


@dataclass
class CloudArcherState:
    """A ranged enemy riding a cloud, reusing ArcherState (and so
    render.draw.draw_archer, spawn_arrow, head/torso/limb_hitbox) directly
    for the character itself -- it already has everything a bow-wielder
    needs (aim_angle, draw, reload_timer). Only the id, hover behavior, and
    per-instance accuracy/cadence live on this wrapper, mirroring
    MeleeEnemyState's split between shared render/hitbox state and AI state.
    """

    archer_id: int
    state: ArcherState
    fire_cooldown: float = field(default_factory=lambda: 0.0)
    aim_noise: float = CLOUD_ARCHER_AIM_NOISE
