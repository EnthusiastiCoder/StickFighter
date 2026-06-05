"""Stands still and never fires. The trivial opponent used before real bots
land, and the floor of the difficulty ladder."""

from stickfighter.core.actions import NOOP, Action
from stickfighter.core.match import MatchSnapshot


class IdleBot:
    def act(self, snapshot: MatchSnapshot, perspective: int) -> Action:
        return NOOP

    def reset(self) -> None:
        pass
