"""Per-step reward shaping for one perspective.

Round outcomes (not raw deaths) carry the win/loss signal, since RoundOver
already correctly captures every way a round ends -- a kill, or a timeout
decided by HP -- with no risk of double-counting a kill as both a death
penalty and a separate win/loss term.
"""

import math

from stickfighter.core import constants as C
from stickfighter.core.events import ArrowHit, Event, RoundOver
from stickfighter.core.match import MatchSnapshot

STEP_PENALTY = -0.001
ROUND_WIN_REWARD = 1.0
ROUND_LOSS_PENALTY = -1.0
DAMAGE_SCALE = 0.5 / 100.0  # +/-0.5 reward per 100 damage dealt/taken
HEADSHOT_BONUS = 0.15  # on top of the round win a kill already causes

NEAR_MISS_RADIUS = 60.0  # px; an own arrow within this of the enemy gets a shaping bonus
NEAR_MISS_MAX_BONUS = 0.02  # per step, at radius == 0 and scale == 1.0
