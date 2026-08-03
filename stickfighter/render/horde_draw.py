"""Rendering for horde-mode melee enemies. Reuses draw.py's leg-cycle helper
(melee enemies walk on the same ground plane as an archer) but replaces the
bow/arms with a weapon read at a glance: a short blade for swordsmen, a long
line for spearmen's reach, and a shield block in front of shieldbearers.

Weapon-arm swing and the forward lean follow standard run-cycle principles
(researched, not guessed): the arm swings opposite the leading leg and
swings higher/faster than a walk; the whole torso leans into the direction
of travel, scaling with effort, while the head stays aligned with the spine.
"""

import math

import pygame

from stickfighter.core import constants as C
from stickfighter.core.horde_entities import MeleeEnemyState
from stickfighter.render import theme
from stickfighter.render.draw import (
    HEAD_RADIUS,
    LINE_WIDTH,
    VISUAL_HIP_Y,
    WALK_FREQUENCY,
    WALK_SWING,
    _circle,
    _draw_legs,
    _leg_points,
    _line,
)

SWORD_LENGTH = 18.0
SPEAR_LENGTH = 34.0
SHIELD_WIDTH = 10.0
SHIELD_HEIGHT = 34.0
ARM_SWING = WALK_SWING * 1.3  # runs swing arms higher/faster than a walk
LEAN_MAX = 9.0  # forward torso lean at full speed, pivoting from the hip
HAND_RADIUS = 2.5


def draw_melee_enemy(
    surface: pygame.Surface, enemy: MeleeEnemyState, elapsed: float, scale: float = 1.0
) -> None:
    if not enemy.alive:
        return
    # a purely render-side silhouette scale (1.0 for humans, bigger for a
    # dragon, smaller for a hound) -- the hitbox stays the standard human
    # size/position (see EnemyConfig.visual_size), so this only stretches
    # the *drawing* upward from the real, unscaled feet position.
    size = enemy.config.visual_size
    feet_y = enemy.y
    top = feet_y - C.CHAR_HEIGHT * size
    hip = (enemy.x, top + VISUAL_HIP_Y * size)

    speed_frac = min(abs(enemy.vx) / C.CHAR_MOVE_SPEED, 1.0)
    # lean the torso into the run, pivoting from the hip -- peaks at full
    # speed, vanishes when stopped (e.g. mid melee-range attack)
    lean = enemy.facing * LEAN_MAX * speed_frac
    head = (enemy.x + lean, top + C.HEAD_CENTER_Y * size)
    shoulder = (enemy.x + lean * 0.6, top + (C.TORSO_TOP_Y + 4.0) * size)

    phase = elapsed * WALK_FREQUENCY
    leg_swing = math.sin(phase) * WALK_SWING * speed_frac if speed_frac > 0.05 else 0.0

    _draw_legs(surface, enemy, hip, feet_y, elapsed, scale)
    _draw_arms(surface, enemy, shoulder, leg_swing, speed_frac, scale, size)

    _line(surface, scale, theme.INK, shoulder, hip, LINE_WIDTH * size)
    _circle(surface, scale, head, HEAD_RADIUS * size, theme.INK)

    if enemy.config.has_shield:
        _draw_shield(surface, enemy, shoulder, hip, scale, size)


def _draw_arms(
    surface: pygame.Surface,
    enemy: MeleeEnemyState,
    shoulder: tuple[float, float],
    leg_swing: float,
    speed_frac: float,
    scale: float,
    size: float = 1.0,
) -> None:
    """Both arms, swinging opposite each other like a real run -- one hand
    forward, one back -- with the weapon staying in only the sword hand.
    Previously only the sword arm was drawn at all, with nothing for the
    off hand."""
    # opposite the leading leg (leg_swing>0 means the *other* leg leads),
    # and swings noticeably harder than the leg cycle -- per run-cycle
    # reference, arms move higher/faster than legs in a run, not a walk.
    arm_swing = -leg_swing * (ARM_SWING / WALK_SWING) if WALK_SWING else 0.0
    forward_lift = (6.0 + 6.0 * speed_frac) * size

    sword_hand = (
        shoulder[0] + enemy.facing * forward_lift + arm_swing,
        shoulder[1] + 10.0 * size - abs(arm_swing) * 0.3,
    )
    length = (SPEAR_LENGTH if enemy.config.name == "spearman" else SWORD_LENGTH) * size
    # the blade continues the shoulder->hand line instead of always pointing
    # in the facing direction -- previously the tip was hand + facing*length
    # unconditionally, so the hand swung back and forth but the blade itself
    # always pointed forward no matter what, never trailing behind the body
    # on the backswing the way a real swung weapon does.
    fx, fy = sword_hand[0] - shoulder[0], sword_hand[1] - shoulder[1]
    arm_len = math.hypot(fx, fy) or 1.0
    tip = (sword_hand[0] + fx / arm_len * length, sword_hand[1] + fy / arm_len * length)

    # a separate, slightly higher "far shoulder" for the off arm -- with
    # both arms hung off the exact same point and the same height, the two
    # overlapped into one indistinct blob whenever their reach coincided
    # (confirmed by rendering and zooming into a frame where that happened).
    far_shoulder = (shoulder[0], shoulder[1] - 6.0 * size)
    off_hand = (
        far_shoulder[0] + enemy.facing * forward_lift - arm_swing,
        far_shoulder[1] + 10.0 * size - abs(arm_swing) * 0.3,
    )

    _line(surface, scale, theme.INK, far_shoulder, off_hand, LINE_WIDTH * size)
    _circle(surface, scale, off_hand, HAND_RADIUS * size, theme.INK)

    _line(surface, scale, theme.INK, shoulder, sword_hand, LINE_WIDTH * size)
    _circle(surface, scale, sword_hand, HAND_RADIUS * size, theme.INK)
    _line(surface, scale, theme.SILVER, sword_hand, tip, max(1, round(3 * size)))


CLOUD_PUFF_RADII = (24.0, 20.0, 17.0, 15.0)
CLOUD_PUFF_OFFSETS = ((-27.0, 3.0), (0.0, -6.0), (27.0, 3.0), (48.0, 12.0))


def draw_cloud(surface: pygame.Surface, x: float, y: float, scale: float) -> None:
    """A few overlapping circles beneath a cloud archer's feet -- enough to
    read as a cloud at a glance without needing a dedicated sprite. Borderless
    on purpose (a prior ink-outline pass was removed per user request) --
    a cloud is soft-edged, unlike every ink-outlined stick figure."""
    for (dx, dy), r in zip(CLOUD_PUFF_OFFSETS, CLOUD_PUFF_RADII):
        _circle(surface, scale, (x + dx, y + dy), r, theme.WHITE)


LEG_MIST_WIDTH = 30.0
LEG_MIST_BLEED = 14.0  # bleeds a bit past the feet into the cloud below
LEG_MIST_MAX_ALPHA = 235
_leg_mist_cache: dict[tuple[float, int], pygame.Surface] = {}


def _leg_mist_gradient(fade_height: float, scale: float) -> pygame.Surface:
    """A vertical alpha gradient, transparent at the top and nearly opaque
    white at the bottom -- cached per (height, scale) since it's the same
    surface every call at a given render resolution."""
    key = (fade_height, round(scale * 100))
    cached = _leg_mist_cache.get(key)
    if cached is not None:
        return cached
    w_px = max(1, round(LEG_MIST_WIDTH * scale))
    h_px = max(1, round(fade_height * scale))
    grad = pygame.Surface((w_px, h_px), pygame.SRCALPHA)
    for row in range(h_px):
        alpha = round(LEG_MIST_MAX_ALPHA * (row / max(1, h_px - 1)))
        pygame.draw.line(grad, (*theme.WHITE, alpha), (0, row), (w_px, row))
    _leg_mist_cache[key] = grad
    return grad


def draw_leg_mist(
    surface: pygame.Surface, x: float, hip_y: float, feet_y: float, scale: float
) -> None:
    """Fades a cloud archer's lower legs into soft white mist instead of
    ending in a hard silhouette edge, reading as the legs sinking into the
    cloud they stand on (per user request)."""
    fade_top = hip_y + (feet_y - hip_y) * 0.45
    fade_height = feet_y - fade_top + LEG_MIST_BLEED
    grad = _leg_mist_gradient(fade_height, scale)
    rect = grad.get_rect()
    rect.centerx = round(x * scale)
    rect.top = round(fade_top * scale)
    surface.blit(grad, rect)


def _draw_shield(
    surface: pygame.Surface,
    enemy: MeleeEnemyState,
    shoulder: tuple[float, float],
    hip: tuple[float, float],
    scale: float,
    size: float = 1.0,
) -> None:
    cx = enemy.x + enemy.facing * (C.CHAR_HALF_WIDTH + SHIELD_WIDTH * size / 2.0)
    cy = (shoulder[1] + hip[1]) / 2.0
    w, h = round(SHIELD_WIDTH * size * scale), round(SHIELD_HEIGHT * size * scale)
    rect = pygame.Rect(0, 0, w, h)
    rect.center = (round(cx * scale), round(cy * scale))
    border = max(1, round(3 * scale))
    outline_width = max(1, round(2 * scale))
    pygame.draw.rect(surface, theme.WOOD_BROWN, rect, border_radius=border)
    pygame.draw.rect(surface, theme.INK, rect, width=outline_width, border_radius=border)
