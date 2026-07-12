"""Difficulty ladder for human play: scripted bot presets for Bronze/Silver/
Gold, and a real trained checkpoint for Legend whenever one exists on disk
-- the game's hardest difficulty literally *is* the AI training result,
falling back to a perfect-play scripted bot if no checkpoint has been
trained yet.
"""

from collections.abc import Callable
from pathlib import Path

from stickfighter.ai.bots.ballistic import EASY, HARD, MEDIUM, PERFECT, BallisticBot
from stickfighter.ai.bots.base import Policy
