"""HordeSim: a hardcoded hero archer vs. waves of hardcoded melee enemies and
cloud archers, closing in from both sides of a center-stage hero.

Not a game mode and not RL -- a standalone deterministic scenario built to
render one highlight video. The hero is fully autonomous (PERFECT-tier
ballistic aim, no reaction delay -- except a brief forced pivot when it needs
to engage the opposite side from its current facing, see TURN_DURATION), so
there is no external Action input at all: reset(seed) + repeated step() is
the whole interface.

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
    CLOUD_ARCHER_HEIGHT_RANGE,
    CLOUD_ARCHER_HP,
    CLOUD_ARCHER_INITIAL_COOLDOWN,
    CLOUD_ARCHER_STANDOFF,
    SHIELDBEARER,
    SPEARMAN,
    SPEED_VARIANCE,
    SWORDSMAN,
    CloudArcherState,
    MeleeEnemyState,
)
from stickfighter.core.rules import coins_for_part, damage_for_part

HERO_X = C.ARENA_WIDTH / 2.0  # hero stands center-stage, horde closes from both sides
# Death.who has no entity-kind field, and melee enemy_id/archer_id are both
# assigned starting from 0 independently -- offsetting archer ids keeps the
# two ranges from colliding, so a renderer can tell a melee death from an
# archer death by checking `who` against this threshold (see
# game/record_horde.py, which gives melee deaths a fade instead of the
# shared ragdoll).
ARCHER_ID_OFFSET = 100_000
TOTAL_ENEMIES = 60
ARCHER_TOTAL = 40  # 60:40 melee:archer ratio, per user request (100 kills total)
MAX_ACTIVE = 4
MAX_ACTIVE_ARCHERS = 1  # 2 active: 24/30 seeds cleared; 1 active: 29/30, at the 60:40 ratio
ATTACK_COOLDOWN = 0.6
ENEMY_MIX = ((SWORDSMAN, 0.5), (SPEARMAN, 0.3), (SHIELDBEARER, 0.2))
# Originally tuned (seed-swept) with HERO_X near the left edge (90px), where
# a right-side spawn's travel distance to the hero was ARENA_WIDTH + jitter
# - 90 (~955px average). Moving the hero to center stage roughly halved that
# runway (~565px) without changing enemy speed or the hero's fire cooldown
# at all -- confirmed directly by measurement (a reliability sweep collapsed
# from 29/30 clears to 0/20 once the hero was centered). Shifting the jitter
# range by the same +390px the hero moved restores the original ~955px
# runway on both sides, a spawn-distance correction for the new geometry,
# not a pacing/difficulty change.
_HERO_SHIFT = HERO_X - 90.0
SPAWN_X_JITTER = (20.0 + _HERO_SHIFT, 150.0 + _HERO_SHIFT)
# NOT shifted by _HERO_SHIFT like the melee range above: an archer's key
# timing constraint is travel distance to *on-screen* (0..ARENA_WIDTH, an
# absolute frame, not hero-relative -- see _update_archers' on_screen gate,
# which archers must cross before ever firing), not travel distance to the
# hero. Shifting this too once made archers spawn far enough out (up to
# ~1570px from center) that the hero -- which has no on-screen restriction
# on what it can snipe -- reliably killed every single one while still
# off-screen and still on cooldown from its own first shot: confirmed
# directly, 0 archer_fire events in an entire 100-kill run despite 40
# archer kills.
ARCHER_SPAWN_X_JITTER = (60.0, 220.0)
# See the on_screen gate in _update_archers. Must stay under
# C.OUT_OF_BOUNDS_MARGIN (80px) -- firing from further out than that spawns
# an arrow that's already past the arrow-despawn line at the moment it's
# created, killing it on the very same tick it's fired. Confirmed directly:
# an archer at x=-142.6 fired an arrow that resolved as an immediate
# "out_of_bounds" miss on the same tick, with the arrow never appearing in
# even a single rendered frame.
CLOUD_ARCHER_ONSCREEN_MARGIN = 60.0
# The hero's own fire cadence, decoupled from C.RELOAD_TIME (shared with the
# real duel game -- changing that would speed up every archer everywhere).
# At C.RELOAD_TIME (0.45s) the hero clears the "nearest" threat almost as
# fast as anything can close in, regardless of how fast enemies move --
# confirmed by measuring actual kill distances (300-500px from the hero,
# nowhere near melee_range 45-65px) even after nearly doubling enemy speed.
# A slower, horde-specific cadence plus more simultaneously active enemies
# (MAX_ACTIVE) is what actually lets a wave close real distance.
# Faster than the original 0.5s -- per user request ("make the hero a bit
# faster... fire rate") after measuring that stretching the fight to 100
# kills at the original cadence made every seed eventually die to compounded
# bad luck (0/200+ cleared; see DODGE_SPEED and TURN_DURATION below, tuned
# together for the same reason). Eased back up from an initial 0.3s per
# follow-up feedback ("firing speed is a bit too fast") -- 0.35 still clears
# a real majority of seeds (12/20 measured), just not as many as 0.3 (17/20).
HERO_FIRE_COOLDOWN = 0.35
# A brief snap-turn when the hero's next shot is on the opposite side from
# its current facing -- per user request, enemies now approach from both
# sides, so the hero can't always be facing the right way. A real
# vulnerability window (no firing while turning), not just cosmetic.
# Shortened from an initial 0.25s per user request ("reduce the turning
# time").
TURN_DURATION = 0.1

# Hero dodge: the hero used to be fully stationary (HERO_X fixed forever),
# which only worked because nothing could ever hit it from range. Cloud
# archers change that -- the hero now roams within DODGE_RANGE, strafing
# (and sometimes jumping) away from an arrow it predicts will otherwise
# connect, then drifting back toward HERO_X when nothing is imminent.
# Symmetric around HERO_X now that threats close from both sides (it used
# to be lopsided -- (HERO_X-55, HERO_X+230) -- back when everything only
# ever approached from the right).
DODGE_RANGE = (HERO_X - 150.0, HERO_X + 150.0)
DODGE_SPEED = 380.0  # faster strafe, per user request ("make the hero... faster movement")
DODGE_REACT_WINDOW = (0.12, 0.5)  # seconds-to-impact that counts as "react now"
DODGE_DANGER_BAND = 42.0  # px from hero's torso center that counts as "on course"
HOMING_DRIFT_SPEED = 140.0  # easing back toward HERO_X when not dodging
# A melee enemy this close (well before its own melee_range of 45-65) jumps
# the targeting queue ahead of cloud archers -- see _pick_target. 150px
# wasn't enough buffer: at the faster enemy speeds, a melee enemy can cross
# more than 150px in one hero fire-cooldown cycle, so it could still cross
# into real melee_range before the hero's next shot. Swept this against 30
# seeds (150px: 4 hero deaths; 220px: 3) before landing here. Checked as an
# absolute distance now that a melee enemy can close in from either side.
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


def _even_sides(rng: random.Random, count: int) -> list:
    """count spawn sides, split as evenly as possible between -1 (spawns off
    the right, closes in facing left) and +1 (mirror image off the left),
    shuffled so the two sides interleave over the run instead of one side
    draining before the other starts."""
    half = count // 2
    sides = [-1] * half + [1] * (count - half)
    rng.shuffle(sides)
    return sides


@dataclass
class HordeSim:
    rng: random.Random = field(default_factory=random.Random)
    hero: ArcherState = field(default_factory=lambda: ArcherState(x=HERO_X, y=C.GROUND_Y))
    enemies: list[MeleeEnemyState] = field(default_factory=list)
    archers: list[CloudArcherState] = field(default_factory=list)
    arrows: list[ArrowState] = field(default_factory=list)
    _queue: list = field(default_factory=list)  # (EnemyConfig, spawn_side) pairs
    _archer_sides: list = field(default_factory=list)  # spawn_side per remaining archer
    _next_arrow_id: int = 0
    _next_enemy_id: int = 0
    _next_archer_id: int = 0
    total_kills: int = 0
    cleared: bool = False
    tick: int = 0
    log: list[dict] = field(default_factory=list)
    _claimed_targets: dict = field(default_factory=dict)  # arrow_id -> claimed target key
    _dodge_dir: float = 0.0
    _dodge_timer: float = 0.0
    _turn_timer: float = 0.0
    _turning_to: int = 0

    def reset(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.hero = ArcherState(x=HERO_X, y=C.GROUND_Y, facing=1, grounded=True)
        self.enemies = []
        self.archers = []
        self.arrows = []
        # spawn_side -1 means "spawns off the right edge, facing left, closing
        # in toward negative x" and +1 the mirror image -- an even split,
        # shuffled so the two sides interleave rather than draining one side
        # first (see _spawn_up_to_max).
        self._queue = list(
            zip(
                (_weighted_choice(self.rng, ENEMY_MIX) for _ in range(TOTAL_ENEMIES)),
                _even_sides(self.rng, TOTAL_ENEMIES),
            )
        )
        self._archer_sides = _even_sides(self.rng, ARCHER_TOTAL)
        self._next_arrow_id = 0
        self._next_enemy_id = 0
        self._next_archer_id = ARCHER_ID_OFFSET
        self.total_kills = 0
        self.cleared = False
        self.tick = 0
        self.log = []
        self._claimed_targets = {}
        self._dodge_dir = 0.0
        self._dodge_timer = 0.0
        self._turn_timer = 0.0
        self._turning_to = 0
        self._spawn_up_to_max()

    def _spawn_up_to_max(self) -> None:
        active = sum(1 for e in self.enemies if e.alive)
        while active < MAX_ACTIVE and self._queue:
            config, side = self._queue.pop(0)
            x = (
                C.ARENA_WIDTH + self.rng.uniform(*SPAWN_X_JITTER)
                if side < 0
                else -self.rng.uniform(*SPAWN_X_JITTER)
            )
            self.enemies.append(
                MeleeEnemyState(
                    enemy_id=self._next_enemy_id,
                    config=config,
                    x=x,
                    y=C.GROUND_Y,
                    hp=config.hp,
                    facing=side,
                    speed_multiplier=self.rng.uniform(*SPEED_VARIANCE),
                )
            )
            self._next_enemy_id += 1
            active += 1

        active_archers = sum(1 for a in self.archers if a.state.alive)
        while active_archers < MAX_ACTIVE_ARCHERS and self._archer_sides:
            side = self._archer_sides.pop(0)
            x = (
                C.ARENA_WIDTH + self.rng.uniform(*ARCHER_SPAWN_X_JITTER)
                if side < 0
                else -self.rng.uniform(*ARCHER_SPAWN_X_JITTER)
            )
            height = self.rng.uniform(*CLOUD_ARCHER_HEIGHT_RANGE)
            state = ArcherState(
                x=x,
                y=C.GROUND_Y - height,
                facing=side,
                grounded=True,  # airborne legs read as "falling"; hovering is its own pose
                hp=CLOUD_ARCHER_HP,
            )
            self.archers.append(
                CloudArcherState(
                    archer_id=self._next_archer_id,
                    state=state,
                    fire_cooldown=self.rng.uniform(*CLOUD_ARCHER_INITIAL_COOLDOWN),
                )
            )
            self._next_archer_id += 1
            active_archers += 1

    def step(self) -> list[Event]:
        events: list[Event] = []
        if self.cleared or not self.hero.alive:
            return events

        self._update_hero_dodge()
        # enemies/archers move (and set their real vx) before the hero
        # picks a lead point off of it -- a freshly spawned entity defaults
        # to vx=0.0 until its first _update_enemies/_update_archers call, so
        # firing before that ran meant the hero's lead prediction treated a
        # target that was about to start closing fast as stationary.
        # Confirmed directly: most misses were long-range shots at a
        # target_vx of exactly 0.0 landing far past where that target had
        # actually moved to by the time the (slow, arcing) arrow arrived.
        self._update_enemies(events)
        self._update_archers(events)
        self._act_hero(events)
        self._update_arrows(events)
        self._spawn_up_to_max()

        total_target = TOTAL_ENEMIES + ARCHER_TOTAL
        if (
            self.total_kills >= total_target
            and not self._queue
            and not self._archer_sides
            and not any(e.alive for e in self.enemies)
            and not any(a.state.alive for a in self.archers)
        ):
            self.cleared = True
        self.tick += 1
        return events

    def _update_hero_dodge(self) -> None:
        """React to enemy (cloud archer) arrows on a collision course:
        strafe away (and, conditionally, jump -- see below), then drift back
        toward HERO_X once nothing is imminent. The hero used to be
        permanently anchored at HERO_X, which only worked because nothing
        could ever hit it from range -- cloud archers change that.

        Jump is conditional, not a coin flip: an earlier version jumped on
        50% of dodges regardless of the shot, and tracing an actual death
        showed why that's wrong in general -- for a steeply lobbed,
        nearly-vertically-falling arrow (the norm at these ranges/arc
        angles, threatening the torso/head), jumping moves the hero's head
        *up* into the descending arrow's path instead of away from it
        (confirmed from the per-shot log: hero.y climbing from 500 to 408
        while the fatal arrow fell from y=193 to y=347 -- they met in the
        middle). But a low, near-flat shot heading for the *legs* is the
        opposite case -- jumping lifts the whole body up and clear of it --
        so jump is now used only when the predicted impact point is below
        the torso (see threat_is_low).

        Archers now fire from both sides, so "on a collision course" can no
        longer assume vx<0 / arrow.x>hero.x (that was only ever true for a
        right-side archer) -- time-to-reach-hero.x is computed directly from
        vx's actual sign instead."""
        threat_time: float | None = None
        threat_is_low = False
        leg_line_y = self.hero.top() + C.TORSO_BOTTOM_Y
        for arrow in self.arrows:
            if arrow.owner == 0 or arrow.vx == 0.0:
                continue
            t = (self.hero.x - arrow.x) / arrow.vx
            if t <= 0.0 or not (DODGE_REACT_WINDOW[0] <= t <= DODGE_REACT_WINDOW[1]):
                continue
            future_y = arrow.y + arrow.vy * t + 0.5 * C.GRAVITY * t * t
            hero_center_y = self.hero.top() + (C.TORSO_TOP_Y + C.TORSO_BOTTOM_Y) / 2.0
            if abs(future_y - hero_center_y) > DODGE_DANGER_BAND:
                continue
            if threat_time is None or t < threat_time:
                threat_time = t
                threat_is_low = future_y > leg_line_y

        if threat_time is not None and self._dodge_timer <= 0.0:
            self._dodge_dir = self.rng.choice((-1.0, 1.0))
            self._dodge_timer = 0.35
            if threat_is_low and self.hero.grounded:
                self.hero.vy = -C.CHAR_JUMP_SPEED
                self.hero.grounded = False
        self._dodge_timer = max(0.0, self._dodge_timer - C.DT)

        if self._dodge_timer > 0.0:
            self.hero.vx = self._dodge_dir * DODGE_SPEED
        elif self.hero.x < HERO_X - 2.0:
            self.hero.vx = HOMING_DRIFT_SPEED
        elif self.hero.x > HERO_X + 2.0:
            self.hero.vx = -HOMING_DRIFT_SPEED
        else:
            self.hero.vx = 0.0

        self.hero.x += self.hero.vx * C.DT
        self.hero.x = max(DODGE_RANGE[0], min(DODGE_RANGE[1], self.hero.x))

        if not self.hero.grounded:
            self.hero.vy += C.GRAVITY * C.DT
            self.hero.y += self.hero.vy * C.DT
            if self.hero.y >= C.GROUND_Y:
                self.hero.y = C.GROUND_Y
                self.hero.vy = 0.0
                self.hero.grounded = True

    def _pick_targets(self) -> list:
        """Ordered candidates, most to least preferred: an urgent unclaimed
        melee enemy (close enough to be about to reach melee_range) beats
        everything -- always prioritizing cloud archers let melee damage
        stack up unanswered, confirmed as the actual cause of hero deaths
        that had no hero_hit log entry at all (i.e. died to melee, not
        arrows). Then unclaimed cloud archers by distance, then unclaimed
        melee by distance, then anything alive at all. Returns a list of
        (tier, "archer"|"melee", target, target_key): tier is the index of
        the preference bucket above (0 = urgent melee), which _act_hero uses
        to decide whether a mismatched-side candidate is worth pivoting for
        right now versus just falling through to the next candidate."""
        archers_alive = [a for a in self.archers if a.state.alive]
        melee_alive = [e for e in self.enemies if e.alive]
        claimed = set(self._claimed_targets.values())

        unclaimed_melee = sorted(
            (e for e in melee_alive if ("melee", e.enemy_id) not in claimed),
            key=lambda e: abs(e.x - self.hero.x),
        )
        urgent_melee = [e for e in unclaimed_melee if abs(e.x - self.hero.x) < URGENT_MELEE_GAP]
        unclaimed_archers = sorted(
            (a for a in archers_alive if ("archer", a.archer_id) not in claimed),
            key=lambda a: abs(a.state.x - self.hero.x),
        )

        def mk(tier, kind, obj):
            key = ("archer", obj.archer_id) if kind == "archer" else ("melee", obj.enemy_id)
            return (tier, kind, obj, key)

        candidates = (
            [mk(0, "melee", e) for e in urgent_melee]
            + [mk(1, "archer", a) for a in unclaimed_archers]
            + [mk(2, "melee", e) for e in unclaimed_melee]
            + sorted(
                [mk(3, "archer", a) for a in archers_alive]
                + [mk(3, "melee", e) for e in melee_alive],
                key=lambda c: abs((c[2].state.x if c[1] == "archer" else c[2].x) - self.hero.x),
            )
        )
        # de-dupe while preserving order (a target can appear in multiple tiers)
        seen = set()
        ordered = []
        for c in candidates:
            if c[3] in seen:
                continue
            seen.add(c[3])
            ordered.append(c)
        return ordered

    def _aim_point(self, kind: str, target) -> tuple[float, float, float, str]:
        if kind == "archer":
            hb = torso_hitbox(target.state)
            tx, ty = (hb.left + hb.right) / 2.0, (hb.top + hb.bottom) / 2.0
            return tx, ty, target.state.vx, "cloud_archer"
        if target.config.has_shield:
            hb = head_hitbox(target)
            tx, ty = hb.cx, hb.cy
        else:
            hb = torso_hitbox(target)
            tx, ty = (hb.left + hb.right) / 2.0, (hb.top + hb.bottom) / 2.0
        return tx, ty, target.vx, target.config.name

    def _act_hero(self, events: list[Event]) -> None:
        self.hero.reload_timer = max(0.0, self.hero.reload_timer - C.DT)

        if self._turn_timer > 0.0:
            # committed to a pivot -- see the "needs a turn" branches below.
            # Deliberately uninterruptible (doesn't re-evaluate candidates
            # mid-turn): simplest rule that still gives turning a real cost,
            # and avoids flip-flopping between two urgent targets on
            # opposite sides before either turn completes.
            self._turn_timer = max(0.0, self._turn_timer - C.DT)
            self.hero.draw = 0.0
            if self._turn_timer <= 0.0:
                self.hero.facing = self._turning_to
            else:
                return

        candidates = self._pick_targets()
        # nock -> draw -> release, not an instant snap to full draw: the
        # string visibly pulls back over the reload cooldown, reaching full
        # draw right as the hero becomes ready to fire again. Previously
        # hero.draw jumped straight to 1.0 the instant a target existed and
        # back to 0.0 the instant it fired, so the render (which does
        # interpolate string pull off hero.draw, see draw.py's STRING_PULL_*)
        # never actually showed a draw happening -- confirmed by user
        # feedback that the animation skipped straight from empty to fired.
        draw_progress = 1.0 - self.hero.reload_timer / HERO_FIRE_COOLDOWN
        self.hero.draw = max(0.0, min(1.0, draw_progress)) if candidates else 0.0

        if not candidates:
            return

        bow_x = self.hero.x + self.hero.facing * C.BOW_FORWARD_OFFSET
        bow_y = self.hero.top() + C.BOW_HEIGHT_OFFSET

        chosen = None
        turn_to: int | None = None  # highest-priority mismatched-side pick seen so far
        for tier, kind, target, target_key in candidates:
            if tier == 3:
                # tier 3 is "everything is already claimed by an in-flight
                # arrow, but track the nearest thing anyway" -- tiers 0-2
                # already cover every *unclaimed* target, so reaching tier 3
                # means firing here would just be a second arrow at
                # something already being shot. With a faster fire cooldown
                # than flight time at long range, that used to happen
                # constantly: confirmed directly, 346/448 fired arrows
                # missing (77%), almost all "out_of_bounds" well past a
                # target that had already died to an earlier, still-in-
                # flight arrow. Skipped entirely here (never fires, never
                # triggers a turn); still usable as the final aim-only
                # fallback below via candidates[0] if nothing else exists.
                continue
            target_x = target.state.x if kind == "archer" else target.x
            desired_facing = 1 if target_x >= self.hero.x else -1
            if desired_facing != self.hero.facing:
                if tier == 0:
                    # urgent melee: can't afford to keep shooting the current
                    # side while this closes in unanswered -- that was the
                    # original reason this tier exists (see _pick_targets).
                    # Turn now instead of finishing off the current side.
                    self._turning_to = desired_facing
                    self._turn_timer = TURN_DURATION
                    self.hero.draw = 0.0
                    return
                if turn_to is None:
                    turn_to = desired_facing
                # not urgent enough to interrupt a still-valid current-side
                # shot for -- keep scanning for one before paying the turn
                # cost. Without this, two roughly-matched threats on
                # opposite sides made the hero pivot back and forth every
                # TURN_DURATION and never actually fire (confirmed: 0/20
                # seeds cleared, hero often dying without a single shot in
                # the back half of a run).
                continue
            tx, ty, target_vx, target_type = self._aim_point(kind, target)
            # lead prediction: aim where the (still-moving) target will be
            # when the arrow arrives, not where it is now -- same
            # one-iteration time-of-flight approximation bots/ballistic.py
            # uses, refined once below.
            dist = math.hypot(tx - bow_x, ty - bow_y)
            time_of_flight = dist / C.ARROW_SPEED_MAX
            tx_led = tx + target_vx * time_of_flight
            dx, dy = tx_led - bow_x, ty - bow_y
            angle = solve_launch_angle(dx, dy, C.ARROW_SPEED_MAX, C.GRAVITY, prefer_low=True)
            if angle is None:
                angle = solve_launch_angle(dx, dy, C.ARROW_SPEED_MAX, C.GRAVITY, prefer_low=False)
            if angle is not None:
                # refine: dist/ARROW_SPEED_MAX is a straight-line lower bound
                # on flight time, only accurate for a near-flat shot. A
                # lobbed shot's horizontal speed is ARROW_SPEED_MAX*cos(angle)
                # -- noticeably less at long range's steeper angles -- so the
                # first pass under-leads a moving target and the arrow flies
                # past where it used to be. Confirmed directly: shots at
                # long-spawn (~900px) targets missed 77% of the time before
                # this refinement, all landing out_of_bounds well past the
                # target's lead point, versus a clean hit rate up close.
                # Recomputing the lead from this angle's actual horizontal
                # speed and re-solving once converges enough to fix it.
                horizontal_speed = C.ARROW_SPEED_MAX * math.cos(angle)
                if abs(horizontal_speed) > 1.0:
                    time_of_flight = abs(dx / horizontal_speed)
                    tx_led = tx + target_vx * time_of_flight
                    dx, dy = tx_led - bow_x, ty - bow_y
                    refined = solve_launch_angle(
                        dx, dy, C.ARROW_SPEED_MAX, C.GRAVITY, prefer_low=True
                    )
                    if refined is None:
                        refined = solve_launch_angle(
                            dx, dy, C.ARROW_SPEED_MAX, C.GRAVITY, prefer_low=False
                        )
                    if refined is not None:
                        angle = refined
                # an arrow despawns as "out of bounds" past the arena edge
                # plus a small margin (core/rules.py's arrow lifetime, shared
                # with the real duel game) -- a target spawned beyond that
                # line is a real, in-range ballistic solution that can never
                # actually connect, since the arrow gets deleted before it
                # ever travels far enough to arrive. Confirmed directly by
                # tracing one such shot: the target spawned at x=1370, the
                # solver happily returned an angle for it, and the arrow was
                # culled at x=1040 (960 + the 80px margin) while the target
                # was still sitting at x=1287 -- nowhere near where the
                # arrow died. Skipping these and waiting for the target to
                # close into reach is the same "don't fire a doomed shot"
                # policy solve_launch_angle's None case already gets below.
                oob = C.OUT_OF_BOUNDS_MARGIN
                if not (-oob <= tx_led <= C.ARENA_WIDTH + oob):
                    continue
                chosen = {
                    "kind": kind, "target": target, "key": target_key, "angle": angle,
                    "type": target_type, "x": target_x, "vx": target_vx, "led_x": tx_led,
                }
                break

        if chosen is None:
            if turn_to is not None:
                # nothing reachable on the current side at all -- pivot
                # toward the highest-priority candidate on the other one.
                self._turning_to = turn_to
                self._turn_timer = TURN_DURATION
                self.hero.draw = 0.0
                return
            # every candidate matched the current facing but none are in
            # range yet -- confirmed real (not a fallback bug): a cloud
            # archer can spawn well beyond the bow's actual max range
            # (speed^2/gravity, less for an elevated target), and the old
            # atan2-with-no-gravity fallback fired a doomed shot every time
            # this happened instead of just waiting.
            _tier, kind, target, _key = candidates[0]
            _tx, ty, _vx, _ = self._aim_point(kind, target)
            tgt_x = target.state.x if kind == "archer" else target.x
            self.hero.aim_angle = math.atan2(ty - bow_y, tgt_x - bow_x)
            return

        self.hero.aim_angle = chosen["angle"]

        ready = self.hero.reload_timer <= 0.0 and self.hero.draw >= C.MIN_DRAW
        if ready:
            arrow = spawn_arrow(self.hero, owner=0, arrow_id=self._next_arrow_id)
            self._next_arrow_id += 1
            self.arrows.append(arrow)
            self._claimed_targets[arrow.id] = chosen["key"]
            self.hero.reload_timer = HERO_FIRE_COOLDOWN
            self.hero.draw = 0.0
            events.append(ArrowFired(owner=0, x=arrow.x, y=arrow.y, vx=arrow.vx, vy=arrow.vy))
            self.log.append({
                "tick": self.tick, "event": "fire", "arrow_id": arrow.id,
                "target_kind": chosen["kind"], "target_type": chosen["type"],
                "target_x": chosen["x"], "target_vx": chosen["vx"],
                "predicted_target_x": chosen["led_x"], "aim_angle": chosen["angle"],
            })

    def _update_enemies(self, events: list[Event]) -> None:
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            gap = abs(enemy.x - self.hero.x)
            if gap > enemy.config.melee_range:
                # facing is fixed at spawn to point toward the hero's side
                # (-1 for a right-spawn closing in leftward, +1 for a
                # left-spawn closing in rightward) -- see _spawn_up_to_max.
                enemy.vx = enemy.facing * enemy.speed()
                enemy.x += enemy.vx * C.DT
                enemy.attack_cooldown = 0.0
            else:
                enemy.vx = 0.0
                enemy.attack_cooldown -= C.DT
                if enemy.attack_cooldown <= 0.0:
                    enemy.attack_cooldown = ATTACK_COOLDOWN
                    self._damage_hero(enemy.config.melee_damage, headshot=False, events=events)

    def _update_archers(self, events: list[Event]) -> None:
        for cloud in self.archers:
            archer = cloud.state
            if not archer.alive:
                continue
            gap = abs(archer.x - self.hero.x)
            standoff_lo, standoff_hi = CLOUD_ARCHER_STANDOFF
            # archer.facing is fixed at spawn (see _spawn_up_to_max) to point
            # toward the hero's side, so facing*DRIFT closes the gap and
            # -facing*DRIFT opens it, regardless of which side this archer
            # spawned on.
            if gap > standoff_hi:
                archer.vx = archer.facing * CLOUD_ARCHER_DRIFT_SPEED
            elif gap < standoff_lo:
                archer.vx = -archer.facing * CLOUD_ARCHER_DRIFT_SPEED  # too close -- back off
            else:
                archer.vx = 0.0
            archer.x += archer.vx * C.DT

            cloud.fire_cooldown -= C.DT
            # don't open fire until nearly on screen -- confirmed by
            # rendering: firing from spawn (which used to be 1000+px beyond
            # the visible arena) meant the very first shots came from an
            # archer nobody could see yet, at the outer edge of the bow's
            # actual range. A small margin beyond the strict edge (rather
            # than requiring x in [0, ARENA_WIDTH] exactly) is still visibly
            # close -- the fired arrow itself is what the viewer tracks, and
            # it's onscreen within a fraction of a second -- and matters now
            # that ARCHER_SPAWN_X_JITTER is much smaller than the old
            # far-off-screen spawns: without slack here, the hero (with no
            # such restriction on what it can snipe) reliably killed every
            # archer while still just off-screen and on cooldown from its
            # first shot, confirmed directly (0 archer_fire events across an
            # entire run despite 40 archer kills).
            on_screen = -CLOUD_ARCHER_ONSCREEN_MARGIN <= archer.x <= (
                C.ARENA_WIDTH + CLOUD_ARCHER_ONSCREEN_MARGIN
            )
            if cloud.fire_cooldown <= 0.0 and on_screen:
                cloud.fire_cooldown = self.rng.uniform(*CLOUD_ARCHER_FIRE_COOLDOWN)
                self._archer_fire(cloud, events)

    def _archer_fire(self, cloud: CloudArcherState, events: list[Event]) -> None:
        archer = cloud.state
        hb = torso_hitbox(self.hero)
        tx, ty = (hb.left + hb.right) / 2.0, (hb.top + hb.bottom) / 2.0
        bow_x = archer.x + archer.facing * C.BOW_FORWARD_OFFSET
        bow_y = archer.top() + C.BOW_HEIGHT_OFFSET

        archer.draw = 0.72  # a notch under full hero-tier power -- slower, dodgeable arcs
        speed = C.ARROW_SPEED_MIN + archer.draw * (C.ARROW_SPEED_MAX - C.ARROW_SPEED_MIN)
        dist = math.hypot(tx - bow_x, ty - bow_y)
        time_of_flight = dist / speed
        tx += self.hero.vx * time_of_flight
        dx, dy = tx - bow_x, ty - bow_y
        angle = solve_launch_angle(dx, dy, speed, C.GRAVITY, prefer_low=True)
        if angle is None:
            angle = solve_launch_angle(dx, dy, speed, C.GRAVITY, prefer_low=False)
        if angle is None:
            angle = math.atan2(dy, dx)
        angle += self.rng.gauss(0.0, cloud.aim_noise)
        archer.aim_angle = angle

        arrow = spawn_arrow(archer, owner=1, arrow_id=self._next_arrow_id)
        self._next_arrow_id += 1
        self.arrows.append(arrow)
        archer.draw = 0.0
        events.append(ArrowFired(owner=1, x=arrow.x, y=arrow.y, vx=arrow.vx, vy=arrow.vy))
        self.log.append({
            "tick": self.tick, "event": "archer_fire", "arrow_id": arrow.id,
            "archer_id": cloud.archer_id, "hero_x": self.hero.x, "aim_angle": angle,
        })

    def _damage_hero(self, damage: float, headshot: bool, events: list[Event]) -> None:
        self.hero.hp = max(0.0, self.hero.hp - damage)
        if self.hero.hp <= 0.0 and self.hero.alive:
            self.hero.alive = False
            events.append(
                Death(
                    who=0, killer=-1, headshot=headshot,
                    x=self.hero.x, y=self.hero.y, facing=self.hero.facing,
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
        if arrow.owner == 0:
            if self._resolve_against_enemies(arrow, prev, cur, events):
                return True
        elif self.hero.alive:
            for part, hitbox, test in (
                ("head", head_hitbox(self.hero), segment_hits_circle),
                ("torso", torso_hitbox(self.hero), segment_hits_rect),
                ("limb", limb_hitbox(self.hero), segment_hits_rect),
            ):
                point = test(prev, cur, hitbox, C.ARROW_COLLIDE_RADIUS)
                if point is None:
                    continue
                damage = damage_for_part(part)
                self.log.append({
                    "tick": self.tick, "event": "hero_hit", "arrow_id": arrow.id,
                    "part": part, "damage": damage, "point": point,
                })
                events.append(
                    ArrowHit(
                        owner=1, target=0, part=part, damage=damage, x=point[0], y=point[1]
                    )
                )
                self._damage_hero(damage, headshot=(part == "head"), events=events)
                return True

        if arrow.y >= C.GROUND_Y:
            angle = math.atan2(arrow.vy, arrow.vx)
            events.append(ArrowMiss(owner=arrow.owner, x=arrow.x, y=C.GROUND_Y, angle=angle))
            self.log.append({
                "tick": self.tick, "event": "miss", "arrow_id": arrow.id,
                "reason": "ground", "point": (arrow.x, C.GROUND_Y),
            })
            return True
        m = C.OUT_OF_BOUNDS_MARGIN
        if arrow.x < -m or arrow.x > C.ARENA_WIDTH + m or arrow.y < -m:
            events.append(ArrowMiss(owner=arrow.owner, x=arrow.x, y=arrow.y))
            self.log.append({
                "tick": self.tick, "event": "miss", "arrow_id": arrow.id,
                "reason": "out_of_bounds", "point": (arrow.x, arrow.y),
            })
            return True
        return False

    def _resolve_against_enemies(
        self,
        arrow: ArrowState,
        prev: tuple[float, float],
        cur: tuple[float, float],
        events: list[Event],
    ) -> bool:
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
                self._apply_hit_melee(enemy, part, point, events)
                return True

        for cloud in self.archers:
            if not cloud.state.alive:
                continue
            for part, hitbox, test in (
                ("head", head_hitbox(cloud.state), segment_hits_circle),
                ("torso", torso_hitbox(cloud.state), segment_hits_rect),
                ("limb", limb_hitbox(cloud.state), segment_hits_rect),
            ):
                point = test(prev, cur, hitbox, C.ARROW_COLLIDE_RADIUS)
                if point is None:
                    continue
                self._apply_hit_archer(cloud, part, point, events)
                return True
        return False

    def _apply_hit_melee(
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

    def _apply_hit_archer(
        self, cloud: CloudArcherState, part: str, point: tuple[float, float], events: list[Event]
    ) -> None:
        archer = cloud.state
        damage = damage_for_part(part)
        archer.hp = max(0.0, archer.hp - damage)
        events.append(
            ArrowHit(
                owner=0, target=cloud.archer_id, part=part, damage=damage, x=point[0], y=point[1]
            )
        )
        events.append(CoinEarned(who=0, amount=coins_for_part(part)))
        killed = archer.hp <= 0.0 and archer.alive
        self.log.append({
            "tick": self.tick, "event": "archer_hit", "target_id": cloud.archer_id,
            "part": part, "damage": damage, "point": point, "killed": killed,
        })
        if killed:
            archer.alive = False
            self.total_kills += 1
            events.append(
                Death(
                    who=cloud.archer_id,
                    killer=0,
                    headshot=(part == "head"),
                    x=archer.x,
                    y=archer.y,
                    facing=archer.facing,
                )
            )

    def snapshot(self) -> HordeSnapshot:
        return HordeSnapshot(
            hero=self.hero,
            enemies=tuple(e for e in self.enemies if e.alive),
            archers=tuple(a for a in self.archers if a.state.alive),
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
