"""Curriculum: S0 static practice -> S1 noisy bot -> S2 sharp bot -> S3
self-play league, auto-advancing on a rolling win-rate threshold. The
near-miss reward bonus is annealed to zero by S2, since a well-aimed miss
stops being useful shaping signal once the agent can reliably land hits.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
