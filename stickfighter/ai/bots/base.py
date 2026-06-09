"""The Policy protocol every bot, and later the trained network, implements."""

from typing import Protocol

from stickfighter.core.actions import Action
from stickfighter.core.match import MatchSnapshot


class Policy(Protocol):
    def act(self, snapshot: MatchSnapshot, perspective: int) -> Action: ...

    def reset(self, seed: int | None = None) -> None:
        """Called at the start of every episode. A bot with internal
        randomness must reseed it here when `seed` is given, or two episodes
        started from the same env seed will not behave identically."""
        ...
