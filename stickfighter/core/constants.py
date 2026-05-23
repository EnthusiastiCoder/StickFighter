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

# Bow / arrows. Draw fraction (0..1) maps linearly onto arrow launch speed.
RELOAD_TIME = 0.45  # seconds of cooldown after releasing before another shot can fire
MIN_DRAW = 0.15  # draw fractions below this do not release a shot
ARROW_SPEED_MIN = 380.0  # px/s at draw == MIN_DRAW
ARROW_SPEED_MAX = 980.0  # px/s at draw == 1.0
ARROW_LENGTH = 34.0
ARROW_COLLIDE_RADIUS = 4.0  # thickness used for swept-segment collision tests
BOW_FORWARD_OFFSET = 18.0  # arrow spawns this far ahead of the archer, along facing
BOW_HEIGHT_OFFSET = 24.0  # arrow spawns this far above the archer's feet

# Hitbox layout, measured downward from the top of the character's bounding box
# (top = feet_y - CHAR_HEIGHT). A headshot is an instant kill; torso/limb chip HP.
HEAD_RADIUS = 9.0
HEAD_CENTER_Y = 9.0
TORSO_TOP_Y = 18.0
TORSO_BOTTOM_Y = 46.0
LIMB_BOTTOM_Y = CHAR_HEIGHT
