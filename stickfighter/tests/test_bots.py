import math

from stickfighter.ai.bots.ballistic import HARD, BallisticBot
from stickfighter.ai.bots.idle import IdleBot
from stickfighter.ai.bots.wanderer import WandererBot
from stickfighter.core.actions import NOOP
from stickfighter.core.arena import ARENAS
from stickfighter.core.match import MatchSim

_FLAT = next(a for a in ARENAS if a.name == "flat")
