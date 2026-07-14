import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

pygame.init()
pygame.display.set_mode((1, 1))

from stickfighter.game.difficulty import DIFFICULTIES  # noqa: E402
from stickfighter.game.scenes.duel import DuelScene  # noqa: E402
from stickfighter.game.scenes.mode_select import ModeSelectScene  # noqa: E402
