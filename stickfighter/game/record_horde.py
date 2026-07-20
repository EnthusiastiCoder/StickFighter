"""Renders a full HordeSim run to a video file: the hero-vs-melee-waves
highlight scenario, not a playable mode. Same "cheap sim trains, pretty
renderer records" split as evaluate.py::record_match, reusing the existing
EffectsSystem/RagdollManager unchanged since HordeSim emits the same Event
types core/match.py does.
"""

import argparse
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio
import pygame

from stickfighter.core import constants as C
from stickfighter.core.horde import TOTAL_ENEMIES, HordeSim
from stickfighter.render import effects, ragdoll, theme
from stickfighter.render.draw import draw_arrow
from stickfighter.render.horde_draw import draw_melee_enemy

_ARCHER_DRAW = None  # lazy import to avoid a hard pygame dependency at module load for tests


def _draw_hero(surface, hero, elapsed, victory: bool = False) -> None:
    global _ARCHER_DRAW
    if _ARCHER_DRAW is None:
        from stickfighter.render.draw import draw_archer

        _ARCHER_DRAW = draw_archer
    _ARCHER_DRAW(surface, hero, elapsed, victory=victory)


def _write_log(out_path: str, log: list[dict]) -> str:
    log_path = os.path.splitext(out_path)[0] + "_log.jsonl"
    with open(log_path, "w") as f:
        for entry in log:
            f.write(json.dumps(entry) + "\n")
    return log_path


def _draw_kill_counter(surface: pygame.Surface, kills: int, font: pygame.font.Font) -> None:
    text = font.render(f"KILLS: {kills}/{TOTAL_ENEMIES}", True, theme.INK)
    surface.blit(text, (16, 12))


def record(out_path: str, seed: int = 0, fps: int = 60, max_steps: int = 20_000) -> dict:
    pygame.init()
    font = pygame.font.Font(None, 32)
    surface = pygame.Surface((int(C.ARENA_WIDTH), int(C.ARENA_HEIGHT)))

    sim = HordeSim()
    sim.reset(seed=seed)
    fx = effects.EffectsSystem()
    ragdolls = ragdoll.RagdollManager()
    elapsed = 0.0

    with imageio.get_writer(out_path, fps=fps) as writer:
        steps = 0
        while not sim.cleared and sim.hero.alive and steps < max_steps:
            events = sim.step()
            elapsed += C.DT
            steps += 1
            fx.handle_events(events)
            ragdolls.handle_events(events)
            fx.update(C.DT, sim.snapshot().arrows)
            ragdolls.update(C.DT)
