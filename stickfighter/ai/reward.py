"""Per-step reward shaping for one perspective.

Round outcomes (not raw deaths) carry the win/loss signal, since RoundOver
already correctly captures every way a round ends -- a kill, or a timeout
decided by HP -- with no risk of double-counting a kill as both a death
penalty and a separate win/loss term.
"""

import math
