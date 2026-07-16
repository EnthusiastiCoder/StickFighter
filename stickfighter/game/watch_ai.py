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
