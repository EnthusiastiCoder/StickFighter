"""Stick-figure and bow/arrow rendering. Pure drawing: reads ArcherState /
ArrowState, never mutates them."""

import math

import pygame

from stickfighter.core import constants as C
from stickfighter.core.arena import Arena
from stickfighter.core.ballistics import bow_origin
from stickfighter.core.entities import ArcherState, ArrowState
from stickfighter.render import theme

PLATFORM_THICKNESS = 14.0

LINE_WIDTH = 5
HEAD_RADIUS = C.HEAD_RADIUS + 1
