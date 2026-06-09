"""Gymnasium environment for 1v1 duels: the learning agent controls one
archer against a configurable opponent Policy (a scripted bot during
curriculum stages, another network snapshot during self-play).

Frame-skip repeats the agent's action for `frame_skip` sim steps (60Hz sim ->
30Hz decisions by default), which is plenty of control resolution for this
game and roughly doubles throughput. The opponent is re-queried every sim
step regardless of frame_skip, so it stays fully reactive.
"""

from typing import Any

import gymnasium as gym
import numpy as np

from stickfighter.ai.bots.base import Policy
from stickfighter.ai.bots.idle import IdleBot
from stickfighter.ai.obs import OBS_DIM, encode_obs
from stickfighter.ai.reward import compute_reward
from stickfighter.core.actions import Action
from stickfighter.core.arena import Arena
from stickfighter.core.events import Event
from stickfighter.core.match import MatchSim

DEFAULT_FRAME_SKIP = 2
DEFAULT_MAX_EPISODE_STEPS = 3600  # 60s of sim time at 60Hz


class DuelEnv(gym.Env):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        opponent: Policy | None = None,
        perspective: int = 0,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        near_miss_scale: float = 0.0,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        arena: Arena | None = None,
    ) -> None:
        super().__init__()
        self.sim = MatchSim()
        self.opponent: Policy = opponent if opponent is not None else IdleBot()
        self.perspective = perspective
        self.frame_skip = frame_skip
        self.near_miss_scale = near_miss_scale
        self.max_episode_steps = max_episode_steps
        self._fixed_arena = arena
        self._elapsed_steps = 0

        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.sim.reset(seed=seed, arena=self._fixed_arena)
        self.opponent.reset(seed=seed)
        self._elapsed_steps = 0
        obs = encode_obs(self.sim.snapshot(), self.perspective)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        my_action = Action.from_array(np.asarray(action, dtype=np.float32))
        all_events: list[Event] = []

        for _ in range(self.frame_skip):
            opp_action = self.opponent.act(self.sim.snapshot(), 1 - self.perspective)
            actions = (
                (my_action, opp_action) if self.perspective == 0 else (opp_action, my_action)
            )
            all_events.extend(self.sim.step(*actions))
            self._elapsed_steps += 1
            if self.sim.match_over:
                break
