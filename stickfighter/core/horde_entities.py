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
    # Render-only silhouette scale (see horde_draw.py's draw_melee_enemy) --
    # deliberately does NOT touch the hitbox, which stays the standard human
    # size/position for every type. A bigger/smaller *look* was the explicit
    # ask ("scaled human silhouette"), not bigger/smaller hittable area.
    visual_size: float = 1.0


SWORDSMAN = EnemyConfig(
    name="swordsman", hp=30.0, speed=320.0, melee_range=45.0, melee_damage=25.0, has_shield=False
)
SPEARMAN = EnemyConfig(
    name="spearman", hp=30.0, speed=270.0, melee_range=65.0, melee_damage=30.0, has_shield=False
)
SHIELDBEARER = EnemyConfig(
    name="shieldbearer", hp=30.0, speed=195.0, melee_range=45.0, melee_damage=45.0, has_shield=True
)
# HP is per user request (2x a human's 30). Speed/range/damage weren't
# specified, so this leans into an archetype: a fast, weaker skirmisher
# (quicker than any human, hits for less). visual_size bumped from an
# initial 0.75 per follow-up feedback ("why do the hounds look like small
# dogs? they gotta be bigger") -- a wolf reads as a real threat, not a
# lapdog.
HOUND = EnemyConfig(
    name="hound", hp=60.0, speed=380.0, melee_range=40.0, melee_damage=20.0,
    has_shield=False, visual_size=1.2,
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
# CLOUD_ARCHER_FIRE_COOLDOWN above). Was (0.15, 0.4) -- already short -- then
# tightened to near-zero per user request ("as soon as archers spawn, they
# should try to fire an arrow immediately at the hero, makes it fun").
CLOUD_ARCHER_INITIAL_COOLDOWN = (0.0, 0.05)
CLOUD_ARCHER_AIM_NOISE = 0.10  # radians, gaussian -- real but beatable accuracy

# Dragons are aerial, not ground melee, per user request ("dragons should be
# aerial mobs... fly and have long range breath to deal damage") -- a flying,
# ranged attacker is structurally the same shape as a cloud archer (hover at
# a standoff distance, fire a ranged projectile at the hero), so a dragon is
# a CloudArcherState tagged kind="dragon" rather than a whole separate
# entity type. HP is 4x a human's 30 per the original request; the rest
# leans into "aerial bruiser with a long-range breath": stands off farther
# than a cloud archer, fires less often but hits much harder (see
# DRAGON_BREATH_DAMAGE_MULT, applied the same way as the hero's big arrow --
# ArrowState.damage_mult), and aims more accurately (a breath is harder to
# just dodge than a spread of aim_noise).
DRAGON_HP = 120.0
DRAGON_HEIGHT_RANGE = (140.0, 240.0)  # flies a bit higher than a cloud archer
# A breath is "long range" -- stands back further than a cloud archer's
# (220, 400) -- but capped well under HERO_X (480, center stage) so it
# settles on screen rather than at/past the edge: the initial (320, 520)
# let it stop as far as x=480+520=1000, off the 0..960 visible frame,
# confirmed directly (a whole run never put it below x=959 on screen).
DRAGON_STANDOFF = (250.0, 420.0)
DRAGON_DRIFT_SPEED = 70.0
DRAGON_FIRE_COOLDOWN = (2.6, 3.4)
# Tightened to near-zero for the same reason as CLOUD_ARCHER_INITIAL_COOLDOWN
# above (per user request, an immediate shot on spawn).
DRAGON_INITIAL_COOLDOWN = (0.0, 0.05)
DRAGON_AIM_NOISE = 0.06
DRAGON_BREATH_DAMAGE_MULT = 2.0


@dataclass
class CloudArcherState:
    """A ranged enemy riding a cloud, reusing ArcherState (and so
    render.draw.draw_archer, spawn_arrow, head/torso/limb_hitbox) directly
    for the character itself -- it already has everything a bow-wielder
    needs (aim_angle, draw, reload_timer). Only the id, hover behavior, and
    per-instance accuracy/cadence live on this wrapper, mirroring
    MeleeEnemyState's split between shared render/hitbox state and AI state.

    kind distinguishes a plain cloud archer from a dragon (see above) --
    both fly the same update/fire pipeline (horde.py's _update_archers/
    _archer_fire), just with different stats and a different render shape.
    """

    archer_id: int
    state: ArcherState
    fire_cooldown: float = field(default_factory=lambda: 0.0)
    aim_noise: float = CLOUD_ARCHER_AIM_NOISE
    kind: str = "cloud_archer"
    damage_mult: float = 1.0
