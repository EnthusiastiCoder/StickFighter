"""Death ragdoll: a short-lived pymunk simulation for corpses.

This is purely cosmetic -- MatchSim has already reset for the next round by
the time a ragdoll plays, so nothing here can feed back into gameplay.
"""

import random

import pygame
import pymunk

from stickfighter.core import constants as C
from stickfighter.core.events import Death, Event
from stickfighter.render import theme
