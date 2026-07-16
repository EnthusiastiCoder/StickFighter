"""Standalone launcher: opens a real window straight into AI-vs-AI spectate
mode, skipping the title/mode-select screens, for quickly watching a
specific matchup of trained/scripted difficulties play live.

    py -3.12 -m stickfighter.game.watch_ai --left Legend --right Gold
"""

import argparse

import pygame

from stickfighter.core import constants as C
from stickfighter.game.difficulty import DIFFICULTIES
from stickfighter.game.scenes.duel import DuelScene

MAX_FRAME_DT = 0.25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", choices=list(DIFFICULTIES), default="Legend")
    parser.add_argument("--right", choices=list(DIFFICULTIES), default="Gold")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)
