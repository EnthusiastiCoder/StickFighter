"""The Policy protocol every bot, and later the trained network, implements."""

from typing import Protocol

from stickfighter.core.actions import Action
from stickfighter.core.match import MatchSnapshot


class Policy(Protocol):
    def act(self, snapshot: MatchSnapshot, perspective: int) -> Action: ...

    def reset(self) -> None: ...
