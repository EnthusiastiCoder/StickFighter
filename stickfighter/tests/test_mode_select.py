import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

pygame.init()
pygame.display.set_mode((1, 1))

from stickfighter.game.difficulty import DIFFICULTIES  # noqa: E402
from stickfighter.game.scenes.duel import DuelScene  # noqa: E402
from stickfighter.game.scenes.mode_select import ModeSelectScene  # noqa: E402


def test_has_one_button_per_difficulty_plus_spectate():
    scene = ModeSelectScene()
    assert {name for _rect, name in scene._difficulty_buttons} == set(DIFFICULTIES)
    assert scene._spectate_button is not None
