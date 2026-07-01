"""Curriculum: S0 static practice -> S1 noisy bot -> S2 sharp bot -> S3
self-play league, auto-advancing on a rolling win-rate threshold. The
near-miss reward bonus is annealed to zero by S2, since a well-aimed miss
stops being useful shaping signal once the agent can reliably land hits.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from stickfighter.ai.bots.ballistic import EASY, HARD, BallisticBot
from stickfighter.ai.bots.base import Policy
from stickfighter.ai.bots.idle import IdleBot

SELFPLAY_STAGE_NAME = "S3_selfplay"


@dataclass(frozen=True)
class Stage:
    name: str
    opponent_factory: Callable[[], Policy] | None  # None for the self-play stage
    near_miss_scale: float
    advance_win_rate: float
    min_matches: int  # matches required at this stage before it can advance


def _easy_bot() -> Policy:
    return BallisticBot(EASY)


def _hard_bot() -> Policy:
    return BallisticBot(HARD)


STAGES: tuple[Stage, ...] = (
    Stage("S0_static", IdleBot, near_miss_scale=1.0, advance_win_rate=0.90, min_matches=50),
    Stage("S1_noisy_bot", _easy_bot, near_miss_scale=0.5, advance_win_rate=0.75, min_matches=100),
    Stage("S2_sharp_bot", _hard_bot, near_miss_scale=0.0, advance_win_rate=0.60, min_matches=150),
    Stage(SELFPLAY_STAGE_NAME, None, near_miss_scale=0.0, advance_win_rate=1.1, min_matches=10**9),
)


@dataclass
class Curriculum:
    window: int = 50
    stage_idx: int = 0
    results: list[bool] = field(default_factory=list)
    matches_this_stage: int = 0

    @property
    def stage(self) -> Stage:
        return STAGES[self.stage_idx]

    @property
    def win_rate(self) -> float:
        return sum(self.results) / len(self.results) if self.results else 0.0

    def record_match(self, trainee_won: bool) -> bool:
        """Returns True if the curriculum just advanced to a new stage."""
        self.results.append(trainee_won)
        if len(self.results) > self.window:
            self.results.pop(0)
        self.matches_this_stage += 1
