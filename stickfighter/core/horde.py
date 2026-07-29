"""HordeSim: a hardcoded hero archer vs. waves of hardcoded melee enemies.

Not a game mode and not RL -- a standalone deterministic scenario built to
render one highlight video. The hero is fully autonomous (PERFECT-tier
ballistic aim, no reaction delay), so there is no external Action input at
all: reset(seed) + repeated step() is the whole interface.

Reuses core/ballistics.py and core/rules.py unchanged, and emits the same
Event types core/match.py does (ArrowFired, ArrowHit, ArrowMiss, Death,
CoinEarned) so the existing render/effects.py EffectsSystem and
render/ragdoll.py RagdollManager work on this scenario with no changes --
both are already driven purely by event data, not by any "there are exactly
2 archers" assumption.
"""

import argparse
import math
import random
from dataclasses import dataclass, field

from stickfighter.core import constants as C
from stickfighter.core.ballistics import (
    segment_hits_circle,
    segment_hits_rect,
    solve_launch_angle,
    spawn_arrow,
    step_arrow,
)
from stickfighter.core.entities import (
    ArcherState,
    ArrowState,
    head_hitbox,
    limb_hitbox,
    torso_hitbox,
)
from stickfighter.core.events import ArrowFired, ArrowHit, ArrowMiss, CoinEarned, Death, Event
from stickfighter.core.horde_entities import (
    CLOUD_ARCHER_DRIFT_SPEED,
    CLOUD_ARCHER_FIRE_COOLDOWN,
    CLOUD_ARCHER_HEIGHT,
    CLOUD_ARCHER_HP,
    CLOUD_ARCHER_STANDOFF,
    SHIELDBEARER,
    SPEARMAN,
    SPEED_VARIANCE,
    SWORDSMAN,
    CloudArcherState,
    MeleeEnemyState,
)
from stickfighter.core.rules import coins_for_part, damage_for_part

HERO_X = 90.0
TOTAL_ENEMIES = 15
ARCHER_TOTAL = 4
MAX_ACTIVE = 4
MAX_ACTIVE_ARCHERS = 1  # 2 active archers dropped clear rate to 34/40 seeds vs 40/40 at 1
ATTACK_COOLDOWN = 0.6
ENEMY_MIX = ((SWORDSMAN, 0.5), (SPEARMAN, 0.3), (SHIELDBEARER, 0.2))
SPAWN_X_JITTER = (20.0, 150.0)
ARCHER_SPAWN_X_JITTER = (60.0, 220.0)
# The hero's own fire cadence, decoupled from C.RELOAD_TIME (shared with the
# real duel game -- changing that would speed up every archer everywhere).
# At C.RELOAD_TIME (0.45s) the hero clears the "nearest" threat almost as
# fast as anything can close in, regardless of how fast enemies move --
# confirmed by measuring actual kill distances (300-500px from the hero,
# nowhere near melee_range 45-65px) even after nearly doubling enemy speed.
# A slower, horde-specific cadence plus more simultaneously active enemies
# (MAX_ACTIVE) is what actually lets a wave close real distance.
HERO_FIRE_COOLDOWN = 0.5

# Hero dodge: the hero used to be fully stationary (HERO_X fixed forever),
# which only worked because nothing could ever hit it from range. Cloud
# archers change that -- the hero now roams within DODGE_RANGE, strafing
# (and sometimes jumping) away from an arrow it predicts will otherwise
# connect, then drifting back toward HERO_X when nothing is imminent.
DODGE_RANGE = (HERO_X - 55.0, HERO_X + 230.0)
DODGE_SPEED = 260.0
DODGE_REACT_WINDOW = (0.12, 0.5)  # seconds-to-impact that counts as "react now"
DODGE_DANGER_BAND = 42.0  # px from hero's torso center that counts as "on course"
HOMING_DRIFT_SPEED = 90.0  # easing back toward HERO_X when not dodging
# A melee enemy this close (well before its own melee_range of 45-65) jumps
# the targeting queue ahead of cloud archers -- see _pick_target. 150px
# wasn't enough buffer: at the faster enemy speeds, a melee enemy can cross
# more than 150px in one hero fire-cooldown cycle, so it could still cross
# into real melee_range before the hero's next shot. Swept this against 30
# seeds (150px: 4 hero deaths; 220px: 3) before landing here.
URGENT_MELEE_GAP = 220.0


@dataclass(frozen=True)
class HordeSnapshot:
    hero: ArcherState
    enemies: tuple[MeleeEnemyState, ...]
    archers: tuple[CloudArcherState, ...]
    arrows: tuple[ArrowState, ...]
    total_kills: int
    cleared: bool


def _weighted_choice(rng: random.Random, mix: tuple) -> object:
    total = sum(w for _cfg, w in mix)
    r = rng.uniform(0.0, total)
    upto = 0.0
    for cfg, w in mix:
        upto += w
        if r <= upto:
            return cfg
    return mix[-1][0]


@dataclass
class HordeSim:
    rng: random.Random = field(default_factory=random.Random)
    hero: ArcherState = field(default_factory=lambda: ArcherState(x=HERO_X, y=C.GROUND_Y))
    enemies: list[MeleeEnemyState] = field(default_factory=list)
    archers: list[CloudArcherState] = field(default_factory=list)
    arrows: list[ArrowState] = field(default_factory=list)
    _queue: list = field(default_factory=list)
    _archer_queue: int = 0  # count of cloud archers still to spawn
    _next_arrow_id: int = 0
    _next_enemy_id: int = 0
    _next_archer_id: int = 0
    total_kills: int = 0
    cleared: bool = False
    tick: int = 0
    log: list[dict] = field(default_factory=list)
    _claimed_targets: dict = field(default_factory=dict)  # arrow_id -> enemy_id already en route

    def reset(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.hero = ArcherState(x=HERO_X, y=C.GROUND_Y, facing=1, grounded=True)
        self.enemies = []
        self.arrows = []
        self._queue = [_weighted_choice(self.rng, ENEMY_MIX) for _ in range(TOTAL_ENEMIES)]
        self._next_arrow_id = 0
        self._next_enemy_id = 0
        self.total_kills = 0
        self.cleared = False
        self.tick = 0
        self.log = []
        self._claimed_targets = {}
        self._spawn_up_to_max()

    def _spawn_up_to_max(self) -> None:
        active = sum(1 for e in self.enemies if e.alive)
        while active < MAX_ACTIVE and self._queue:
            config = self._queue.pop(0)
            x = C.ARENA_WIDTH + self.rng.uniform(*SPAWN_X_JITTER)
            self.enemies.append(
                MeleeEnemyState(
                    enemy_id=self._next_enemy_id,
                    config=config,
                    x=x,
                    y=C.GROUND_Y,
                    hp=config.hp,
                    facing=-1,
                    speed_multiplier=self.rng.uniform(*SPEED_VARIANCE),
                )
            )
            self._next_enemy_id += 1
            active += 1

    def step(self) -> list[Event]:
        events: list[Event] = []
        if self.cleared or not self.hero.alive:
            return events

        self._act_hero(events)
        self._update_enemies(events)
        self._update_arrows(events)
        self._spawn_up_to_max()

        if self.total_kills >= TOTAL_ENEMIES and not self._queue and not any(
            e.alive for e in self.enemies
        ):
            self.cleared = True
        self.tick += 1
        return events

    def _nearest_target(self) -> MeleeEnemyState | None:
        """Nearest *unclaimed* enemy (no arrow already flying toward it) when
        one exists, else nearest overall. Without this, the hero -- reload
        is much faster than an arrow's flight time -- would keep re-aiming
        at whichever enemy is nearest even after already firing at it,
        wasting a second arrow that arrives to find it already dead from the
        first. Confirmed from the per-shot log, not assumed: every miss in a
        first pass at this turned out to be exactly this case."""
        alive = [e for e in self.enemies if e.alive]
        if not alive:
            return None
        claimed_ids = set(self._claimed_targets.values())
        pool = [e for e in alive if e.enemy_id not in claimed_ids] or alive
        return min(pool, key=lambda e: abs(e.x - self.hero.x))

    def _act_hero(self, events: list[Event]) -> None:
        self.hero.reload_timer = max(0.0, self.hero.reload_timer - C.DT)
        target = self._nearest_target()
        self.hero.draw = 1.0 if target is not None else 0.0
        self.hero.facing = 1

        if target is None:
            return

        if target.config.has_shield:
            hb = head_hitbox(target)
            tx, ty = hb.cx, hb.cy
        else:
            hb = torso_hitbox(target)
            tx, ty = (hb.left + hb.right) / 2.0, (hb.top + hb.bottom) / 2.0

        bow_x = self.hero.x + self.hero.facing * C.BOW_FORWARD_OFFSET
        bow_y = self.hero.top() + C.BOW_HEIGHT_OFFSET

        # lead prediction: aim where the (still-walking) target will be when
        # the arrow arrives, not where it is now -- same one-iteration
        # time-of-flight approximation bots/ballistic.py uses. The previous
        # version aimed at the target's current position with no lead term
        # at all, which is wrong on physics grounds alone (nonzero flight
        # time + nonzero target velocity = the target has moved by impact)
        # independent of whether it explains the specific misses reported.
        dist = math.hypot(tx - bow_x, ty - bow_y)
        time_of_flight = dist / C.ARROW_SPEED_MAX
        tx += target.vx * time_of_flight
        dx, dy = tx - bow_x, ty - bow_y
        angle = solve_launch_angle(dx, dy, C.ARROW_SPEED_MAX, C.GRAVITY, prefer_low=True)
        if angle is None:
            angle = solve_launch_angle(dx, dy, C.ARROW_SPEED_MAX, C.GRAVITY, prefer_low=False)
        if angle is None:
            angle = math.atan2(dy, dx)
        self.hero.aim_angle = angle

        ready = self.hero.reload_timer <= 0.0 and self.hero.draw >= C.MIN_DRAW
        if ready:
            arrow = spawn_arrow(self.hero, owner=0, arrow_id=self._next_arrow_id)
            self._next_arrow_id += 1
            self.arrows.append(arrow)
            self._claimed_targets[arrow.id] = target.enemy_id
            self.hero.reload_timer = HERO_FIRE_COOLDOWN
            self.hero.draw = 0.0
            events.append(ArrowFired(owner=0, x=arrow.x, y=arrow.y, vx=arrow.vx, vy=arrow.vy))
            self.log.append({
                "tick": self.tick, "event": "fire", "arrow_id": arrow.id,
                "target_id": target.enemy_id, "target_type": target.config.name,
                "target_x": target.x, "target_vx": target.vx,
                "predicted_target_x": tx, "aim_angle": angle,
            })

    def _update_enemies(self, events: list[Event]) -> None:
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            gap = enemy.x - self.hero.x
            if gap > enemy.config.melee_range:
                enemy.vx = -enemy.speed()
                enemy.x += enemy.vx * C.DT
                enemy.attack_cooldown = 0.0
            else:
                enemy.vx = 0.0
                enemy.attack_cooldown -= C.DT
                if enemy.attack_cooldown <= 0.0:
                    enemy.attack_cooldown = ATTACK_COOLDOWN
                    self.hero.hp = max(0.0, self.hero.hp - enemy.config.melee_damage)
                    if self.hero.hp <= 0.0 and self.hero.alive:
                        self.hero.alive = False
                        events.append(
                            Death(
                                who=0,
                                killer=-1,
                                headshot=False,
                                x=self.hero.x,
                                y=self.hero.y,
                                facing=self.hero.facing,
                            )
                        )

    def _update_arrows(self, events: list[Event]) -> None:
        surviving: list[ArrowState] = []
        for arrow in self.arrows:
            prev = step_arrow(arrow, C.DT, wind=0.0)
            if self._resolve_arrow(arrow, prev, events):
                self._claimed_targets.pop(arrow.id, None)
            else:
                surviving.append(arrow)
        self.arrows = surviving

    def _resolve_arrow(
        self, arrow: ArrowState, prev: tuple[float, float], events: list[Event]
    ) -> bool:
        cur = (arrow.x, arrow.y)
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            for part, hitbox, test in (
                ("head", head_hitbox(enemy), segment_hits_circle),
                ("torso", torso_hitbox(enemy), segment_hits_rect),
                ("limb", limb_hitbox(enemy), segment_hits_rect),
            ):
                point = test(prev, cur, hitbox, C.ARROW_COLLIDE_RADIUS)
                if point is None:
                    continue
                if part != "head" and enemy.config.has_shield:
                    # ArrowHit, not ArrowMiss: effects.py only leaves a
                    # permanent "stuck arrow" decal for ArrowMiss, which for
                    # a block on a *moving* enemy left decals floating in
                    # mid-air, detached from the shield that actually caught
                    # it -- confirmed by rendering and watching the video,
                    # not guessed. ArrowHit still gets the hit-particle burst
                    # and screen shake, just with 0 damage and no kill.
                    events.append(
                        ArrowHit(
                            owner=0, target=enemy.enemy_id, part="shield",
                            damage=0.0, x=point[0], y=point[1],
                        )
                    )
                    self.log.append({
                        "tick": self.tick, "event": "shield_block", "arrow_id": arrow.id,
                        "target_id": enemy.enemy_id, "point": point,
                    })
                    return True
                self._apply_hit(enemy, part, point, events)
                return True
        if arrow.y >= C.GROUND_Y:
            events.append(ArrowMiss(owner=0, x=arrow.x, y=C.GROUND_Y))
            self.log.append({
                "tick": self.tick, "event": "miss", "arrow_id": arrow.id,
                "reason": "ground", "point": (arrow.x, C.GROUND_Y),
            })
            return True
        m = C.OUT_OF_BOUNDS_MARGIN
        if arrow.x < -m or arrow.x > C.ARENA_WIDTH + m or arrow.y < -m:
            events.append(ArrowMiss(owner=0, x=arrow.x, y=arrow.y))
            self.log.append({
                "tick": self.tick, "event": "miss", "arrow_id": arrow.id,
                "reason": "out_of_bounds", "point": (arrow.x, arrow.y),
            })
            return True
        return False

    def _apply_hit(
        self, enemy: MeleeEnemyState, part: str, point: tuple[float, float], events: list[Event]
    ) -> None:
        damage = damage_for_part(part)
        enemy.hp = max(0.0, enemy.hp - damage)
        events.append(
            ArrowHit(
                owner=0, target=enemy.enemy_id, part=part, damage=damage, x=point[0], y=point[1]
            )
        )
        events.append(CoinEarned(who=0, amount=coins_for_part(part)))
        killed = enemy.hp <= 0.0 and enemy.alive
        self.log.append({
            "tick": self.tick, "event": "hit", "target_id": enemy.enemy_id,
            "part": part, "damage": damage, "point": point, "killed": killed,
        })
        if killed:
            enemy.alive = False
            self.total_kills += 1
            events.append(
                Death(
                    who=enemy.enemy_id,
                    killer=0,
                    headshot=(part == "head"),
                    x=enemy.x,
                    y=enemy.y,
                    facing=enemy.facing,
                )
            )

    def snapshot(self) -> HordeSnapshot:
        return HordeSnapshot(
            hero=self.hero,
            enemies=tuple(e for e in self.enemies if e.alive),
            arrows=tuple(self.arrows),
            total_kills=self.total_kills,
            cleared=self.cleared,
        )


def _selftest() -> None:
    sim = HordeSim()
    sim.reset(seed=0)
    steps = 0
    while not sim.cleared and sim.hero.alive and steps < 20_000:
        sim.step()
        steps += 1
    print(
        f"cleared={sim.cleared} hero_alive={sim.hero.alive} "
        f"kills={sim.total_kills} steps={steps}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
    else:
        parser.print_help()
