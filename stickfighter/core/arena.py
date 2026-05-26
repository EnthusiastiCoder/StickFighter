"""Arena layouts: the ground plus a handful of floating platforms.

Platforms are one-way: an archer only lands on the top surface while falling
onto it (see character.py); there is no collision from the sides or below.
"""

import random
from dataclasses import dataclass

from stickfighter.core import constants as C
