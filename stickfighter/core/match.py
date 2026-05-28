"""MatchSim: the deterministic, headless duel simulation.

This is the one piece of code that runs identically whether a human is
playing or the AI is training self-play -- there is no separate "fast"
physics model, so nothing can diverge between play and training.
"""

import argparse
import random
from dataclasses import dataclass

from stickfighter.core import constants as C
from stickfighter.core.actions import Action
from stickfighter.core.arena import Arena, random_arena
from stickfighter.core.ballistics import (
    segment_hits_circle,
    segment_hits_rect,
    spawn_arrow,
    step_arrow,
)
from stickfighter.core.character import step_character
from stickfighter.core.entities import (
    ArcherState,
    ArrowState,
    Rect,
    head_hitbox,
    limb_hitbox,
    torso_hitbox,
)
from stickfighter.core.events import (
    ArrowFired,
    ArrowHit,
    ArrowMiss,
    CoinEarned,
    Death,
    Event,
    MatchOver,
    RoundOver,
)
from stickfighter.core.rules import coins_for_part, damage_for_part, is_match_won
