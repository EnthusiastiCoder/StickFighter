"""The playable duel: human archer (P0) vs a placeholder idle opponent (P1).

Real bots (stickfighter.ai.bots) replace the idle opponent once M3 lands --
this scene only needs *an* Action source for P1, not a specific one.
"""

import pygame

from stickfighter.core import constants as C
from stickfighter.core.actions import Action
from stickfighter.core.match import MatchSim
from stickfighter.game.input import HumanInputController
from stickfighter.render import draw, effects, hud, ragdoll, theme
