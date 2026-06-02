"""Frame-to-frame visual effects: arrow trails, floating hit text, screen
shake, blood particles, and stuck-arrow decals. All state here is render-only
and is driven entirely by the Event list MatchSim.step() returns."""

import math
import random
from collections import deque
from dataclasses import dataclass

import pygame

from stickfighter.core import constants as C
from stickfighter.core.entities import ArrowState
from stickfighter.core.events import ArrowHit, ArrowMiss, CoinEarned, Death, Event
from stickfighter.render import theme

TRAIL_LENGTH = 14
TRAIL_DOT_STRIDE = 2
MAX_STUCK_ARROWS = 40


@dataclass
class FloatingText:
    x: float
    y: float
    text: str
    color: tuple[int, int, int]
    age: float = 0.0
    lifetime: float = 0.9
    vy: float = -40.0


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    age: float = 0.0
    lifetime: float = 0.5


@dataclass
class StuckArrow:
    x: float
    y: float


_font_cache: dict[int, pygame.font.Font] = {}


def _get_font(size: int) -> pygame.font.Font:
    if size not in _font_cache:
        _font_cache[size] = pygame.font.Font(None, size)
    return _font_cache[size]


class EffectsSystem:
    def __init__(self) -> None:
        self._trails: dict[int, deque[tuple[float, float]]] = {}
        self.floaters: list[FloatingText] = []
        self.particles: list[Particle] = []
        self.stuck: list[StuckArrow] = []
        self.shake_time = 0.0
        self.shake_magnitude = 0.0
        self._rng = random.Random()

    def handle_events(self, events: list[Event]) -> None:
        last_hit_pos: tuple[float, float] | None = None
        for e in events:
            if isinstance(e, ArrowHit):
                last_hit_pos = (e.x, e.y)
                self._spawn_hit_particles(e.x, e.y)
                self.shake(8.0 if e.part == "head" else 4.0, 0.15)
                if e.part == "head":
                    self.floaters.append(FloatingText(e.x, e.y - 22, "HEADSHOT", theme.GOLD))
            elif isinstance(e, CoinEarned) and last_hit_pos is not None:
                x, y = last_hit_pos
                self.floaters.append(FloatingText(x, y - 6, f"+{e.amount}", theme.GOLD))
            elif isinstance(e, ArrowMiss):
                self.stuck.append(StuckArrow(e.x, e.y))
                if len(self.stuck) > MAX_STUCK_ARROWS:
                    self.stuck.pop(0)
            elif isinstance(e, Death):
                self.shake(10.0, 0.25)

    def _spawn_hit_particles(self, x: float, y: float) -> None:
        for _ in range(8):
            angle = self._rng.uniform(0, math.tau)
            speed = self._rng.uniform(40, 160)
            self.particles.append(Particle(x, y, math.cos(angle) * speed, math.sin(angle) * speed))

    def shake(self, magnitude: float, duration: float) -> None:
        self.shake_magnitude = max(self.shake_magnitude, magnitude)
        self.shake_time = max(self.shake_time, duration)

    def update(self, dt: float, arrows: tuple[ArrowState, ...]) -> None:
        live_ids = {a.id for a in arrows}
        for a in arrows:
            trail = self._trails.setdefault(a.id, deque(maxlen=TRAIL_LENGTH))
            trail.appendleft((a.x, a.y))
        for stale_id in [i for i in self._trails if i not in live_ids]:
            del self._trails[stale_id]

        for p in self.particles:
            p.vy += C.GRAVITY * 0.5 * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.age += dt
        self.particles = [p for p in self.particles if p.age < p.lifetime]

        for f in self.floaters:
            f.y += f.vy * dt
            f.age += dt
        self.floaters = [f for f in self.floaters if f.age < f.lifetime]
