import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

pygame.init()

from stickfighter.ai.bots.ballistic import PERFECT, BallisticBot  # noqa: E402
from stickfighter.ai.bots.idle import IdleBot  # noqa: E402
from stickfighter.core.events import CoinEarned  # noqa: E402
from stickfighter.game import save  # noqa: E402
from stickfighter.game.scenes.duel import DuelScene  # noqa: E402
