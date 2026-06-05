"""Scripted archer that aims with the closed-form ballistics solver. Serves
as both a training-curriculum opponent and a human difficulty ladder --
difficulty is entirely a matter of the noise/delay/target knobs below, not a
different algorithm.
"""

import math
import random
from dataclasses import dataclass

from stickfighter.core import constants as C
from stickfighter.core.actions import Action
from stickfighter.core.ballistics import solve_launch_angle
from stickfighter.core.entities import ArcherState, head_hitbox, torso_hitbox
from stickfighter.core.match import MatchSnapshot


@dataclass(frozen=True)
class BallisticBotConfig:
    aim_noise: float = 0.0  # radians, gaussian stddev added to the solved aim angle
    reaction_delay: float = 0.15  # seconds of hesitation before committing to a shot
    draw_fraction: float = 1.0  # how hard it draws; 1.0 = best range and speed
    lead_prediction: bool = True  # aim where the enemy will be, not where it is now
    target_part: str = "torso"  # "head" or "torso"
    engage_range: tuple[float, float] = (150.0, 900.0)  # (back off below, close in above)


EASY = BallisticBotConfig(
    aim_noise=0.35,
    reaction_delay=0.5,
    draw_fraction=0.85,
    lead_prediction=False,
    target_part="torso",
)
MEDIUM = BallisticBotConfig(
    aim_noise=0.15, reaction_delay=0.3, draw_fraction=0.9, lead_prediction=True, target_part="torso"
)
HARD = BallisticBotConfig(
    aim_noise=0.05, reaction_delay=0.12, draw_fraction=1.0, lead_prediction=True, target_part="head"
)
PERFECT = BallisticBotConfig(
    aim_noise=0.0, reaction_delay=0.0, draw_fraction=1.0, lead_prediction=True, target_part="head"
)


def _speed_for_draw(draw_fraction: float) -> float:
    return C.ARROW_SPEED_MIN + draw_fraction * (C.ARROW_SPEED_MAX - C.ARROW_SPEED_MIN)


class BallisticBot:
    def __init__(self, config: BallisticBotConfig = MEDIUM, seed: int | None = None) -> None:
        self.config = config
        self._rng = random.Random(seed)
        self._hesitation = 0.0

    def reset(self) -> None:
        self._hesitation = 0.0

    def act(self, snapshot: MatchSnapshot, perspective: int) -> Action:
        self_a = snapshot.archers[perspective]
        enemy_a = snapshot.archers[1 - perspective]
        if not self_a.alive:
            return Action(move=0.0, jump=False, aim_angle=self_a.aim_angle, draw=0.0, release=False)

        move = self._movement(enemy_a.x - self_a.x, self.config)
        angle = self._aim(self_a, enemy_a)

        ready = self_a.reload_timer <= 0.0
        self._hesitation = self._hesitation + C.DT if ready else 0.0
        release = ready and self._hesitation >= self.config.reaction_delay

        return Action(
            move=move, jump=False, aim_angle=angle, draw=self.config.draw_fraction, release=release
        )

    def _aim(self, self_a: ArcherState, enemy_a: ArcherState) -> float:
        target_x, target_y = self._target_point(self_a, enemy_a)
        bow_x = self_a.x + self_a.facing * C.BOW_FORWARD_OFFSET
        bow_y = self_a.top() + C.BOW_HEIGHT_OFFSET
        speed = _speed_for_draw(self.config.draw_fraction)
        dx, dy = target_x - bow_x, target_y - bow_y
