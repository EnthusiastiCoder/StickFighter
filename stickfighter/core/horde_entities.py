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
    # -1 walks in from the right (facing left); +1 from the left (facing
    # right) -- fixed at spawn, see horde.py's HordeSim._spawn_up_to_max.
    facing: int = -1
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
# px above the ground the cloud (and archer) hovers -- randomized per-archer
# (see horde.py's spawn) so a wave doesn't fly in at one uniform altitude.
CLOUD_ARCHER_HEIGHT_RANGE = (120.0, 210.0)
# Stops drifting once this close to the hero. Must keep both bounds small
# enough that an archer can settle on-screen from *either* side of a
# center-stage hero (ARENA_WIDTH=960, hero at x=480) -- the old (420, 620)
# was tuned back when the hero stood near the left edge, where a right-side
# archer settling 420-620px out was still comfortably on-screen. Centered,
# the same band let an archer settle as far as hero.x-620 (=-140, off the
# left edge) or hero.x+620 (=1100, off the right edge) and then just sit
# there forever -- confirmed directly: a stalled run had an archer frozen at
# x=-139.9 for 17000+ steps, never close enough to on-screen (x>=0, see
# horde.py's _update_archers) to fire, and "close enough" to stop drifting.
CLOUD_ARCHER_STANDOFF = (220.0, 400.0)
CLOUD_ARCHER_DRIFT_SPEED = 90.0
# Randomized per shot, not a robotic metronome. Slowed from (1.3, 1.9) after
# tightening HERO_ENGAGE_INSET (horde.py) made the hero wait longer to
# engage each archer -- more cooldown cycles pass before it dies, so more
# total shots (and headshot rolls) land per archer. A slower cadence
# partially offsets that: 11/20 seeds cleared at (1.8, 2.4) vs 9/20 at the
# original range, measured at the same engage inset.
CLOUD_ARCHER_FIRE_COOLDOWN = (1.8, 2.4)
# Used only for the very first shot after spawn, not between shots (that's
# CLOUD_ARCHER_FIRE_COOLDOWN above). At the hero's current fire rate and
# targeting priority, an archer starting on the full 1.3-1.9s cooldown got
# sniped before it ever fired a single arrow essentially every time --
# confirmed directly: 0 archer_fire events across an entire 100-kill run
# despite 40 archer kills. A near-immediate first shot gives it a real
# chance to be a threat (and the hero something to dodge) before it dies.
CLOUD_ARCHER_INITIAL_COOLDOWN = (0.15, 0.4)
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
