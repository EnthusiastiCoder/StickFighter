"""Mode/difficulty select: choose a duel difficulty, or watch two AIs fight."""

import pygame

from stickfighter.core import constants as C
from stickfighter.game.difficulty import DIFFICULTIES, make_gold, make_legend
from stickfighter.game.scenes.duel import DuelScene
from stickfighter.render import theme
