"""Stick-figure and bow/arrow rendering. Pure drawing: reads ArcherState /
ArrowState, never mutates them."""

import math

import pygame
import pygame.gfxdraw

from stickfighter.core import constants as C
from stickfighter.core.arena import Arena
from stickfighter.core.ballistics import bow_origin
from stickfighter.core.entities import ArcherState, ArrowState
from stickfighter.render import theme


def draw_aa_circle(
    surface: pygame.Surface, center: tuple[float, float], radius: float, color
) -> None:
    """pygame.draw.circle has no anti-aliasing and looks visibly blocky at
    this game's scale; gfxdraw's filled + outlined circle does."""
    x, y, r = round(center[0]), round(center[1]), round(radius)
    pygame.gfxdraw.filled_circle(surface, x, y, r, color)
    pygame.gfxdraw.aacircle(surface, x, y, r, color)


# -- render-scale support -----------------------------------------------
# All geometry below is computed in the game's native 960x540 coordinate
# space (unchanged math, same as the live 60fps window) and only converted
# to screen pixels at the point of drawing, via these wrappers. This lets a
# caller (record_horde.py) render at e.g. 3x for a genuinely higher-fidelity
# video export -- more source pixels for anti-aliasing to work with, not
# just an upscaled blur of the same small image -- without touching
# core/constants.py, which core/horde.py's *simulation* (hitboxes,
# ballistics) also reads: scaling those globally would silently change
# actual gameplay math, not just how it's drawn.
def _pt(p: tuple[float, float], scale: float) -> tuple[float, float]:
    return (p[0] * scale, p[1] * scale)


def _line(surface, scale: float, color, p0, p1, width: float) -> None:
    pygame.draw.line(surface, color, _pt(p0, scale), _pt(p1, scale), max(1, round(width * scale)))


def _aalines(surface, scale: float, color, points) -> None:
    pygame.draw.aalines(surface, color, False, [_pt(p, scale) for p in points])


def _thick_curve(surface, scale: float, color, points, width: float) -> None:
    """pygame.draw.aalines has no width control -- always ~1px regardless of
    scale, which is wrong for the bow's own wooden limb (should read as the
    bold element; the string is what should be thin). draw.lines does take
    a width but isn't anti-aliased on its own, so oversample: draw at 2x the
    target size onto a small scratch surface with alpha, then smoothscale
    down -- cheap since this is only ever a handful of points."""
    scaled = [_pt(p, scale) for p in points]
    xs = [p[0] for p in scaled]
    ys = [p[1] for p in scaled]
    pad = width * scale + 4
    left, top = min(xs) - pad, min(ys) - pad
    w = max(int(max(xs) - left + pad), 1)
    h = max(int(max(ys) - top + pad), 1)
    ss = 2
    scratch = pygame.Surface((w * ss, h * ss), pygame.SRCALPHA)
    local = [((x - left) * ss, (y - top) * ss) for x, y in scaled]
    pygame.draw.lines(scratch, color, False, local, max(1, round(width * scale * ss)))
    small = pygame.transform.smoothscale(scratch, (w, h))
    surface.blit(small, (left, top))


def _polygon(surface, scale: float, color, points) -> None:
    pygame.draw.polygon(surface, color, [_pt(p, scale) for p in points])


def _circle(surface, scale: float, center, radius: float, color) -> None:
    draw_aa_circle(surface, _pt(center, scale), radius * scale, color)


def _capsule(surface, scale: float, color, p0, p1, r0: float, r1: float) -> None:
    """A solid tapered limb segment -- the core primitive for the game's
    silhouette rig. Every bone (torso, upper-arm, thigh, shin, ...) is drawn
    as one of these instead of a thin line: a trapezoid body plus a round
    cap at each end, so two segments sharing a joint (e.g. thigh meeting
    shin at the knee) blend into one continuous rounded joint instead of a
    hard mitred corner. r0/r1 let a segment taper (e.g. wider at the hip
    than the knee), matching how a real limb narrows toward its extremity."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        _circle(surface, scale, p0, max(r0, r1), color)
        return
    nx, ny = dx / length, dy / length
    perp = (-ny, nx)
    a = (p0[0] + perp[0] * r0, p0[1] + perp[1] * r0)
    b = (p1[0] + perp[0] * r1, p1[1] + perp[1] * r1)
    c = (p1[0] - perp[0] * r1, p1[1] - perp[1] * r1)
    d = (p0[0] - perp[0] * r0, p0[1] - perp[1] * r0)
    _polygon(surface, scale, color, (a, b, c, d))
    _circle(surface, scale, p0, r0, color)
    _circle(surface, scale, p1, r1, color)


PLATFORM_THICKNESS = 14.0

LINE_WIDTH = 5
HEAD_RADIUS = C.HEAD_RADIUS + 1

# Solid-limb radii for the silhouette rig (draw_archer, and reused by
# horde_draw.py's human enemies) -- replaces the old thin LINE_WIDTH stroke
# with a tapered filled capsule per bone, per user request ("our signature
# style is actually silhouettes", not stick figures). Every joint-angle
# computation elsewhere (run cycle, bow draw, sway) is untouched -- only how
# each bone is *drawn* changed, from a stroke to a filled shape.
TORSO_R_TOP = 7.5
TORSO_R_BOTTOM = 5.5
ARM_R_SHOULDER = 5.0
ARM_R_HAND = 3.0
THIGH_R_HIP = 5.5
THIGH_R_KNEE = 4.5
SHIN_R_KNEE = 4.5
SHIN_R_FOOT = 3.0

# Render-only leg proportion: the real torso/limb split (C.TORSO_BOTTOM_Y)
# is also the torso/limb *hitbox* boundary used by the actual duel game's
# collision code, so it can't be moved just to make legs look longer.
# Confirmed by rendering a walking figure and looking: at the real hitbox
# split, legs read as short/stubby (~43% of shoulder-to-feet height). This
# raises the hip point used for drawing only, giving legs more of the
# visible body without touching hitboxes at all.
VISUAL_HIP_Y = C.TORSO_BOTTOM_Y - 10.0

BOW_HALF_LEN = 21.0
BOW_CURVE_BASE = 5.0
BOW_CURVE_DRAW = 6.0
BOW_LIMB_WIDTH = 3.0  # the wooden limb is the bold element
STRING_WIDTH = 1.0  # the string is the thin one -- was drawn thicker than the limb
STRING_PULL_BASE = 4.0
STRING_PULL_MAX = 16.0  # kept well under ARROW_LENGTH (34) so a full draw still
# leaves a visible shaft poking out past the grip toward the target
ARROWHEAD_LEN = 6.0
ARROWHEAD_WIDTH = 3.0
FLETCH_LEN = 6.0  # how far ahead of the tail the fletching vanes start
FLETCH_SPREAD = 5.0  # how far apart the two vanes end up at the tail

WALK_FREQUENCY = 9.0
WALK_SWING = 13.0
IDLE_SWAY_FREQUENCY = 1.6
IDLE_SWAY_AMOUNT = 1.5
RECOIL_WINDOW = 0.12  # seconds after a release the body leans back, for a felt snap


def _aim_vectors(angle: float) -> tuple[tuple[float, float], tuple[float, float]]:
    forward = (math.cos(angle), math.sin(angle))
    perp = (-forward[1], forward[0])
    return forward, perp


def _quad_bezier(p0, p1, p2, segments: int = 10) -> list[tuple[float, float]]:
    pts = []
    for i in range(segments + 1):
        t = i / segments
        mt = 1.0 - t
        x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
        y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
        pts.append((x, y))
    return pts


def _leg_points(
    hip: tuple[float, float], feet_y: float, x_offset: float, bend: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    foot = (hip[0] + x_offset, feet_y)
    knee = ((hip[0] + foot[0]) / 2.0 + bend, (hip[1] + feet_y) / 2.0)
    return hip, knee, foot


def draw_archer(
    surface: pygame.Surface,
    archer: ArcherState,
    elapsed: float,
    victory: bool = False,
    scale: float = 1.0,
    hero: bool = False,
) -> None:
    if not archer.alive:
        return
    top = archer.top()

    idle = archer.grounded and abs(archer.vx) < 1.0 and archer.draw < C.MIN_DRAW
    sway = math.sin(elapsed * IDLE_SWAY_FREQUENCY) * IDLE_SWAY_AMOUNT if idle else 0.0
    recoil_progress = max(0.0, C.RELOAD_TIME - archer.reload_timer) / RECOIL_WINDOW
    recoil = (1.0 - min(recoil_progress, 1.0)) * -6.0 if archer.reload_timer > 0.0 else 0.0

    head = (archer.x - archer.facing * recoil, top + C.HEAD_CENTER_Y + sway)
    shoulder = (archer.x - archer.facing * recoil, top + C.TORSO_TOP_Y + 4.0 + sway)
    hip = (archer.x, top + VISUAL_HIP_Y)
    feet_y = archer.y

    # the cape is drawn first, so its silhouette sits behind the torso/legs
    # instead of overlapping on top of them.
    if hero:
        _draw_cape(surface, archer, shoulder, elapsed, scale)

    _draw_legs(surface, archer, hip, feet_y, elapsed, scale)
    if victory:
        _draw_victory_pose(surface, archer, shoulder, scale)
    else:
        _draw_bow_and_arms(surface, archer, shoulder, scale)

    _capsule(surface, scale, theme.INK, shoulder, hip, TORSO_R_TOP, TORSO_R_BOTTOM)
    _circle(surface, scale, head, HEAD_RADIUS, theme.INK)


CAPE_LENGTH = 34.0
CAPE_SWAY = 10.0
CAPE_FREQUENCY = WALK_FREQUENCY * 0.55  # slower than the leg cycle -- cloth
# lagging behind the body's motion reads as heavier/more valiant than
# something flapping in lockstep with the footsteps.


def _draw_cape(
    surface, archer: ArcherState, shoulder: tuple[float, float], elapsed: float, scale: float
) -> None:
    """The hero only, per user request ("make the hero as valiant as
    possible") -- a flowing cape trailing behind the direction of travel,
    swaying opposite the run cycle so it lags the body like real cloth
    instead of snapping rigidly with each stride."""
    speed_frac = min(abs(archer.vx) / C.CHAR_MOVE_SPEED, 1.0) if archer.grounded else 0.5
    sway = math.sin(elapsed * CAPE_FREQUENCY) * CAPE_SWAY * (0.4 + 0.6 * speed_frac)
    back = -archer.facing
    collar_a = (shoulder[0] + back * 2.0, shoulder[1] - 5.0)
    collar_b = (shoulder[0] + back * 3.0, shoulder[1] + 6.0)
    mid = (shoulder[0] + back * CAPE_LENGTH * 0.55 + sway * 0.5, shoulder[1] + CAPE_LENGTH * 0.35)
    tip_a = (shoulder[0] + back * CAPE_LENGTH + sway, shoulder[1] + CAPE_LENGTH * 0.55)
    tip_b = (
        shoulder[0] + back * CAPE_LENGTH * 0.75 + sway * 0.8,
        shoulder[1] + CAPE_LENGTH * 0.85,
    )
    _polygon(surface, scale, theme.INK, (collar_a, mid, tip_a, tip_b, collar_b))


def _draw_victory_pose(
    surface, archer: ArcherState, shoulder: tuple[float, float], scale: float
) -> None:
    """Bow thrust overhead -- the finishing beat after the last kill."""
    raised = (shoulder[0] + archer.facing * 6.0, shoulder[1] - 34.0)
    _capsule(surface, scale, theme.INK, shoulder, raised, ARM_R_SHOULDER, ARM_R_HAND)
    tip_a = (raised[0] - 10.0, raised[1] - 12.0)
    tip_b = (raised[0] + 10.0, raised[1] - 12.0)
    control = (raised[0], raised[1] - 6.0)
    _aalines(surface, scale, theme.INK, _quad_bezier(tip_a, control, tip_b))


def _draw_legs(
    surface, archer: ArcherState, hip, feet_y: float, elapsed: float, scale: float,
    size: float = 1.0,
) -> None:
    speed_frac = min(abs(archer.vx) / C.CHAR_MOVE_SPEED, 1.0)
    if not archer.grounded:
        legs = (
            _leg_points(hip, feet_y, x_offset=7.0 * archer.facing, bend=4.0 * archer.facing),
            _leg_points(hip, feet_y, x_offset=-9.0 * archer.facing, bend=-5.0 * archer.facing),
        )
    elif speed_frac > 0.05:
        phase = elapsed * WALK_FREQUENCY
        swing = math.sin(phase) * WALK_SWING * speed_frac
        legs = (
            _leg_points(hip, feet_y, x_offset=swing, bend=swing * 0.4),
            _leg_points(hip, feet_y, x_offset=-swing, bend=-swing * 0.4),
        )
    else:
        legs = (
            _leg_points(hip, feet_y, x_offset=6.0, bend=2.0),
            _leg_points(hip, feet_y, x_offset=-6.0, bend=-2.0),
        )
    # size scales limb *thickness* here -- the hip/feet_y passed in by the
    # caller already account for leg *length* (see horde_draw.py's
    # _draw_human), so without this an oversized enemy (e.g. the ogre) grew
    # taller legs that stayed the same thin width, reading as spindly
    # rather than bulky.
    for hip_pt, knee, foot in legs:
        _capsule(surface, scale, theme.INK, hip_pt, knee, THIGH_R_HIP * size, THIGH_R_KNEE * size)
        _capsule(surface, scale, theme.INK, knee, foot, SHIN_R_KNEE * size, SHIN_R_FOOT * size)


def _draw_bow_and_arms(
    surface, archer: ArcherState, shoulder: tuple[float, float], scale: float
) -> None:
    """The two tips and the nock must all rotate together as one rigid body
    around aim_angle -- that is what makes the two string segments
    (tip_a-to-nock, tip_b-to-nock) provably equal length for any aim angle,
    since tip_a/tip_b are then mirror images of each other across the
    forward axis and the nock sits exactly on that axis. Decoupling the
    tips from aim_angle (an earlier version of this function) broke that:
    the string went visibly asymmetric the moment the archer aimed off
    horizontal. The wedge-like look that decoupling was meant to fix was a
    proportions problem (too thin, too small a curve), not a rotation
    problem -- fixed here by making both bigger instead."""
    bow_x, bow_y = bow_origin(archer)
    forward, perp = _aim_vectors(archer.aim_angle)
    draw = archer.draw

    tip_a = (bow_x + perp[0] * BOW_HALF_LEN, bow_y + perp[1] * BOW_HALF_LEN)
    tip_b = (bow_x - perp[0] * BOW_HALF_LEN, bow_y - perp[1] * BOW_HALF_LEN)
    curve = BOW_CURVE_BASE + BOW_CURVE_DRAW * draw
    control = (bow_x + forward[0] * curve, bow_y + forward[1] * curve)
    pull = STRING_PULL_BASE + STRING_PULL_MAX * draw
    nock = (bow_x - forward[0] * pull, bow_y - forward[1] * pull)

    _thick_curve(surface, scale, theme.INK, _quad_bezier(tip_a, control, tip_b), BOW_LIMB_WIDTH)
    _line(surface, scale, theme.INK, tip_a, nock, STRING_WIDTH)
    _line(surface, scale, theme.INK, tip_b, nock, STRING_WIDTH)
    _capsule(surface, scale, theme.INK, shoulder, (bow_x, bow_y), ARM_R_SHOULDER, ARM_R_HAND)
    _capsule(surface, scale, theme.INK, shoulder, nock, ARM_R_SHOULDER, ARM_R_HAND)

    if draw >= C.MIN_DRAW:
        _draw_nocked_arrow(surface, nock, forward, scale)


def _draw_nocked_arrow(
    surface, nock: tuple[float, float], forward: tuple[float, float], scale: float
) -> None:
    tip = (nock[0] + forward[0] * C.ARROW_LENGTH, nock[1] + forward[1] * C.ARROW_LENGTH)
    _line(surface, scale, theme.SILVER, nock, tip, 3)
    perp = (-forward[1], forward[0])
    back = (tip[0] - forward[0] * ARROWHEAD_LEN, tip[1] - forward[1] * ARROWHEAD_LEN)
    head_a = (back[0] + perp[0] * ARROWHEAD_WIDTH, back[1] + perp[1] * ARROWHEAD_WIDTH)
    head_b = (back[0] - perp[0] * ARROWHEAD_WIDTH, back[1] - perp[1] * ARROWHEAD_WIDTH)
    _polygon(surface, scale, theme.INK, (tip, head_a, head_b))


def draw_arena(surface: pygame.Surface, arena: Arena, scale: float = 1.0) -> None:
    for platform in arena.platforms:
        rect = pygame.Rect(
            round(platform.x * scale),
            round(platform.y * scale),
            round(platform.width * scale),
            round(PLATFORM_THICKNESS * scale),
        )
        pygame.draw.rect(surface, theme.INK, rect)


def _arrow_shape(tip: tuple[float, float], dx: float, dy: float, length: float):
    """Shared geometry for both the in-flight arrow and the ground-stuck
    decal, so they always match visually: same silver shaft / black head /
    black fletching, just at different lengths. Returns
    (fletch_start, head_base, head_a, head_b, fletch_a, fletch_b)."""
    tail = (tip[0] - dx * length, tip[1] - dy * length)
    # where the shaft ends and the arrowhead begins: ARROWHEAD_LEN back from
    # the tip, not from the tail -- using the wrong end here previously made
    # the "arrowhead" cover most of the arrow's length instead of a small
    # point at the front, confirmed by rendering a single arrow up close.
    head_base = (tip[0] - dx * ARROWHEAD_LEN, tip[1] - dy * ARROWHEAD_LEN)
    # the shaft only spans from the fletching join to the head -- not all
    # the way back to `tail` -- so nothing pokes out past the fletching's
    # vertex. Drawing it back to `tail` left a bare silver "rod" hanging off
    # the back of the fletching, confirmed by rendering.
    fletch_start = (tail[0] + dx * FLETCH_LEN, tail[1] + dy * FLETCH_LEN)
    perp = (-dy, dx)
    head_a = (head_base[0] + perp[0] * ARROWHEAD_WIDTH, head_base[1] + perp[1] * ARROWHEAD_WIDTH)
    head_b = (head_base[0] - perp[0] * ARROWHEAD_WIDTH, head_base[1] - perp[1] * ARROWHEAD_WIDTH)
    # fletching: two vanes swept back and out from a point *ahead* of the
    # tail to points level with the tail, a shallow V opening backward --
    # not a symmetric crossbar centered on the tail. The crossbar read as a
    # sword's hilt guard once the shaft and head were both silver (same
    # color, similar width), confirmed directly from a rendered frame: a
    # silver in-flight arrow looked like a thrown sword, tip and "guard"
    # both visible, no arrow silhouette at all.
    fletch_a = (tail[0] + perp[0] * FLETCH_SPREAD, tail[1] + perp[1] * FLETCH_SPREAD)
    fletch_b = (tail[0] - perp[0] * FLETCH_SPREAD, tail[1] - perp[1] * FLETCH_SPREAD)
    return fletch_start, head_base, head_a, head_b, fletch_a, fletch_b


def draw_arrow(surface: pygame.Surface, arrow: ArrowState, scale: float = 1.0) -> None:
    speed = math.hypot(arrow.vx, arrow.vy) or 1.0
    dx, dy = arrow.vx / speed, arrow.vy / speed
    tip = (arrow.x, arrow.y)
    fletch_start, head_base, head_a, head_b, fletch_a, fletch_b = _arrow_shape(
        tip, dx, dy, C.ARROW_LENGTH
    )
    _line(surface, scale, theme.SILVER, fletch_start, head_base, 4)
    _polygon(surface, scale, theme.INK, (tip, head_a, head_b))
    _line(surface, scale, theme.INK, fletch_start, fletch_a, 2)
    _line(surface, scale, theme.INK, fletch_start, fletch_b, 2)


# Shorter than a real in-flight arrow (C.ARROW_LENGTH) -- a stuck decal drawn
# at full length buried half of itself past the impact point, looking like
# it pierced deep into the ground, confirmed by rendering. The tip sits
# exactly at the impact point with everything else trailing backward, so
# nothing is drawn on the far side of it at all.
STUCK_ARROW_LEN = 20.0


def draw_stuck_arrow(
    surface: pygame.Surface, x: float, y: float, angle: float, scale: float = 1.0
) -> None:
    # no arrowhead -- per user request, it "looks better" without one on the
    # ground decal (the fletching/"arrowback" stays). The shaft runs all the
    # way to the tip point instead of stopping short at head_base, since
    # there's no separate head shape to hand off to anymore.
    dx, dy = math.cos(angle), math.sin(angle)
    tip = (x, y)
    fletch_start, _head_base, _head_a, _head_b, fletch_a, fletch_b = _arrow_shape(
        tip, dx, dy, STUCK_ARROW_LEN
    )
    _line(surface, scale, theme.SILVER, fletch_start, tip, 3)
    _line(surface, scale, theme.INK, fletch_start, fletch_a, 2)
    _line(surface, scale, theme.INK, fletch_start, fletch_b, 2)
