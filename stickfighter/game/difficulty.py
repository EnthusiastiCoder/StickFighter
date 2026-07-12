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

_CHECKPOINT_CANDIDATES = (
    Path("runs/stickfighter_main/best.pt"),
    Path("runs/stickfighter_main/final.pt"),
)


def find_trained_checkpoint() -> Path | None:
    for candidate in _CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    checkpoints = sorted(runs_dir.glob("*/checkpoints/ckpt_*.pt"))
    return checkpoints[-1] if checkpoints else None


def make_bronze() -> Policy:
    return BallisticBot(EASY)


def make_silver() -> Policy:
    return BallisticBot(MEDIUM)


def make_gold() -> Policy:
    return BallisticBot(HARD)
