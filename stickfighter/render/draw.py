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


def _polygon(surface, scale: float, color, points) -> None:
    pygame.draw.polygon(surface, color, [_pt(p, scale) for p in points])


def _circle(surface, scale: float, center, radius: float, color) -> None:
    draw_aa_circle(surface, _pt(center, scale), radius * scale, color)

PLATFORM_THICKNESS = 14.0

LINE_WIDTH = 5
HEAD_RADIUS = C.HEAD_RADIUS + 1

BOW_HALF_LEN = 21.0
BOW_CURVE_BASE = 5.0
BOW_CURVE_DRAW = 6.0
STRING_PULL_BASE = 4.0
STRING_PULL_MAX = 16.0  # kept well under ARROW_LENGTH (34) so a full draw still
# leaves a visible shaft poking out past the grip toward the target
ARROWHEAD_LEN = 6.0
ARROWHEAD_WIDTH = 3.0

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
    hip = (archer.x, top + C.TORSO_BOTTOM_Y)
    feet_y = archer.y

    _draw_legs(surface, archer, hip, feet_y, elapsed, scale)
    if victory:
        _draw_victory_pose(surface, archer, shoulder)
    else:
        _draw_bow_and_arms(surface, archer, shoulder)

    pygame.draw.line(surface, theme.INK, shoulder, hip, LINE_WIDTH)
    draw_aa_circle(surface, head, HEAD_RADIUS, theme.INK)


def _draw_victory_pose(surface, archer: ArcherState, shoulder: tuple[float, float]) -> None:
    """Bow thrust overhead -- the finishing beat after the last kill."""
    raised = (shoulder[0] + archer.facing * 6.0, shoulder[1] - 34.0)
    pygame.draw.line(surface, theme.INK, shoulder, raised, LINE_WIDTH)
    tip_a = (raised[0] - 10.0, raised[1] - 12.0)
    tip_b = (raised[0] + 10.0, raised[1] - 12.0)
    control = (raised[0], raised[1] - 6.0)
    pygame.draw.aalines(surface, theme.INK, False, _quad_bezier(tip_a, control, tip_b))


def _draw_legs(surface, archer: ArcherState, hip, feet_y: float, elapsed: float) -> None:
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
    for a, b, c in legs:
        pygame.draw.line(surface, theme.INK, a, b, LINE_WIDTH)
        pygame.draw.line(surface, theme.INK, b, c, LINE_WIDTH)


def _draw_bow_and_arms(surface, archer: ArcherState, shoulder: tuple[float, float]) -> None:
    """The bow's own limbs stay a fixed, vertical-ish shape anchored to the
    extended arm (facing direction only) -- rotating the whole silhouette to
    track aim_angle looked fine at a horizontal aim but collapsed into an
    unreadable flat wedge at steep up/down angles (confirmed by rendering
    sample frames, not by inspecting the vector math). Only the string's
    nock point and the arrow itself swing along aim_angle, which is exactly
    what a real draw looks like: the bow stays put in the hand, the string
    hand moves to point at the target."""
    bow_x, bow_y = bow_origin(archer)
    forward, _ = _aim_vectors(archer.aim_angle)
    draw = archer.draw

    tip_a = (bow_x, bow_y - BOW_HALF_LEN)
    tip_b = (bow_x, bow_y + BOW_HALF_LEN)
    curve = BOW_CURVE_BASE + BOW_CURVE_DRAW * draw
    control = (bow_x + archer.facing * curve, bow_y)  # bulges away from the body (the bow's back)
    pull = STRING_PULL_BASE + STRING_PULL_MAX * draw
    nock = (bow_x - forward[0] * pull, bow_y - forward[1] * pull)

    pygame.draw.aalines(surface, theme.INK, False, _quad_bezier(tip_a, control, tip_b))
    pygame.draw.line(surface, theme.INK, tip_a, nock, 2)
    pygame.draw.line(surface, theme.INK, tip_b, nock, 2)
    pygame.draw.line(surface, theme.INK, shoulder, (bow_x, bow_y), LINE_WIDTH)
    pygame.draw.line(surface, theme.INK, shoulder, nock, LINE_WIDTH)

    if draw >= C.MIN_DRAW:
        _draw_nocked_arrow(surface, nock, forward)


def _draw_nocked_arrow(surface, nock: tuple[float, float], forward: tuple[float, float]) -> None:
    tip = (nock[0] + forward[0] * C.ARROW_LENGTH, nock[1] + forward[1] * C.ARROW_LENGTH)
    pygame.draw.line(surface, theme.WHITE, nock, tip, 3)
    perp = (-forward[1], forward[0])
    back = (tip[0] - forward[0] * ARROWHEAD_LEN, tip[1] - forward[1] * ARROWHEAD_LEN)
    head_a = (back[0] + perp[0] * ARROWHEAD_WIDTH, back[1] + perp[1] * ARROWHEAD_WIDTH)
    head_b = (back[0] - perp[0] * ARROWHEAD_WIDTH, back[1] - perp[1] * ARROWHEAD_WIDTH)
    pygame.draw.polygon(surface, theme.INK, (tip, head_a, head_b))


def draw_arena(surface: pygame.Surface, arena: Arena) -> None:
    for platform in arena.platforms:
        rect = pygame.Rect(
            round(platform.x), round(platform.y), round(platform.width), round(PLATFORM_THICKNESS)
        )
        pygame.draw.rect(surface, theme.INK, rect)


def draw_arrow(surface: pygame.Surface, arrow: ArrowState) -> None:
    speed = math.hypot(arrow.vx, arrow.vy) or 1.0
    dx, dy = arrow.vx / speed, arrow.vy / speed
    shaft_len = C.ARROW_LENGTH - ARROWHEAD_LEN
    shaft_back = (arrow.x - dx * shaft_len, arrow.y - dy * shaft_len)
    tail = (arrow.x - dx * C.ARROW_LENGTH, arrow.y - dy * C.ARROW_LENGTH)
    tip = (arrow.x, arrow.y)
    pygame.draw.line(surface, theme.INK, tail, shaft_back, 4)

    perp = (-dy, dx)
    head_a = (shaft_back[0] + perp[0] * ARROWHEAD_WIDTH, shaft_back[1] + perp[1] * ARROWHEAD_WIDTH)
    head_b = (shaft_back[0] - perp[0] * ARROWHEAD_WIDTH, shaft_back[1] - perp[1] * ARROWHEAD_WIDTH)
    pygame.draw.polygon(surface, theme.INK, (tip, head_a, head_b))

    fletch_a = (tail[0] + perp[0] * 5.0, tail[1] + perp[1] * 5.0)
    fletch_b = (tail[0] - perp[0] * 5.0, tail[1] - perp[1] * 5.0)
    pygame.draw.line(surface, theme.INK, tail, fletch_a, 2)
    pygame.draw.line(surface, theme.INK, tail, fletch_b, 2)
