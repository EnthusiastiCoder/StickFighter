"""Frame-to-frame visual effects: arrow trails, floating hit text, screen
shake, blood particles, and stuck-arrow decals. All state here is render-only
and is driven entirely by the Event list MatchSim.step() returns."""

import math
import random
from collections import deque
from dataclasses import dataclass

import pygame

from stickfighter.core import constants as C
from stickfighter.core.entities import ArrowState
from stickfighter.core.events import ArrowHit, ArrowMiss, CoinEarned, Death, Event
from stickfighter.render import theme

TRAIL_LENGTH = 14
TRAIL_DOT_STRIDE = 2
MAX_STUCK_ARROWS = 40
