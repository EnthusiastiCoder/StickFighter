import math

import numpy as np

from stickfighter.ai.obs import OBS_DIM, encode_obs
from stickfighter.core import constants as C
from stickfighter.core.arena import ARENAS
from stickfighter.core.entities import ArcherState
from stickfighter.core.match import MatchSnapshot

_FLAT = next(a for a in ARENAS if a.name == "flat")
