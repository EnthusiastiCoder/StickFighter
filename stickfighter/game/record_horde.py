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
