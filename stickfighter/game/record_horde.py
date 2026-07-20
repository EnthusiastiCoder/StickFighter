"""Renders a full HordeSim run to a video file: the hero-vs-melee-waves
highlight scenario, not a playable mode. Same "cheap sim trains, pretty
renderer records" split as evaluate.py::record_match, reusing the existing
EffectsSystem/RagdollManager unchanged since HordeSim emits the same Event
types core/match.py does.
"""

import argparse
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
