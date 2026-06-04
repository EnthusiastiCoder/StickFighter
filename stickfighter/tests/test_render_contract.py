"""The renderer contract: reading a MatchSnapshot to draw a frame must never
mutate the sim state underneath it (render/ only ever reads)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
