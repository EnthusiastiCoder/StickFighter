"""MatchSim: the deterministic, headless duel simulation.

This is the one piece of code that runs identically whether a human is
playing or the AI is training self-play -- there is no separate "fast"
physics model, so nothing can diverge between play and training.
"""

import argparse
import random
from dataclasses import dataclass
