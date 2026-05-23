"""Simulation constants. No rendering, ML, or I/O dependencies here."""

FPS = 60
DT = 1.0 / FPS

# Arena, in pixels, y-down (y=0 top, increasing downward -- matches pygame/screen space).
ARENA_WIDTH = 960.0
ARENA_HEIGHT = 540.0
GROUND_Y = ARENA_HEIGHT - 40.0
