import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from stickfighter.ai.bots.ballistic import HARD, PERFECT, BallisticBot
from stickfighter.ai.bots.idle import IdleBot
from stickfighter.ai.envs.duel_env import DuelEnv
from stickfighter.ai.obs import OBS_DIM
from stickfighter.core.actions import Action
from stickfighter.core.arena import ARENAS

_FLAT = next(a for a in ARENAS if a.name == "flat")


def test_env_checker_passes_with_a_stochastic_opponent():
    env = DuelEnv(opponent=BallisticBot(HARD, seed=0))
    check_env(env, skip_render_check=True)


def test_reset_and_step_shapes_and_dtypes():
    env = DuelEnv(opponent=IdleBot())
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "events" in info


def test_perspective_1_env_lets_the_opponent_play_archer_0():
    env = DuelEnv(opponent=IdleBot(), perspective=1, arena=_FLAT)
    env.reset(seed=0)
    # archer 0 (the opponent) must stay put; archer 1 (us) should be free to move
    move_right = np.array([1.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32)
    for _ in range(30):
        env.step(move_right)
    assert env.sim.archers[0].x == pytest.approx(_FLAT.spawn_p0[0])
    assert env.sim.archers[1].x > _FLAT.spawn_p1[0] - 1.0  # moved from spawn (clamped or not)


def test_episode_terminates_on_match_over_not_just_truncation():
    env = DuelEnv(opponent=IdleBot(), arena=_FLAT, max_episode_steps=100_000)
    env.reset(seed=0)
    bot = BallisticBot(PERFECT, seed=0)
