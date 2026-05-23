"""Simulation constants. No rendering, ML, or I/O dependencies here."""

FPS = 60
DT = 1.0 / FPS

# Arena, in pixels, y-down (y=0 top, increasing downward -- matches pygame/screen space).
ARENA_WIDTH = 960.0
ARENA_HEIGHT = 540.0
GROUND_Y = ARENA_HEIGHT - 40.0

GRAVITY = 1400.0  # px/s^2, downward. Tuned for ~1-1.5s arrow arcs across the arena.

# Character
CHAR_HALF_WIDTH = 14.0
CHAR_HEIGHT = 64.0
CHAR_MOVE_SPEED = 220.0  # px/s
CHAR_JUMP_SPEED = 520.0  # px/s, upward (negative vy)
CHAR_MAX_FALL_SPEED = 900.0
CHAR_MAX_HP = 100.0
