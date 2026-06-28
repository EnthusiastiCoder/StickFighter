"""Self-play opponent pool: frozen policy snapshots sampled via PFSP
(prioritized fictitious self-play -- weight opponents the trainee is
currently losing to more heavily), plus Elo ratings to track skill
progression across the league.
"""

import copy
import random
from dataclasses import dataclass, field
