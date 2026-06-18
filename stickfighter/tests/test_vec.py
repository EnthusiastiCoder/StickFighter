"""Tests exercise SyncVectorEnv, which runs the exact same _EnvFactory /
DuelEnv construction logic as AsyncVectorEnv, just in-process. AsyncVectorEnv
itself (real multiprocessing workers) was verified manually via a standalone
script under a proper `if __name__ == "__main__":` guard -- the way it will
actually be invoked from ppo/train.py -- since spawning real subprocesses
from inside a pytest run is a separate, flakier concern from the env logic
this test suite is responsible for.
"""

import numpy as np

from stickfighter.ai.bots.idle import IdleBot
from stickfighter.ai.envs.vec import default_opponent_factory, make_vec_env
from stickfighter.ai.obs import OBS_DIM


def test_sync_vec_env_reset_and_step_shapes():
    env = make_vec_env(3, asynchronous=False)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (3, OBS_DIM)

        actions = env.action_space.sample()
        assert actions.shape == (3, 5)
        obs, rewards, terminated, truncated, infos = env.step(actions)
        assert obs.shape == (3, OBS_DIM)
        assert rewards.shape == (3,)
        assert rewards.dtype == np.float64 or rewards.dtype == np.float32
        assert terminated.shape == (3,)
        assert truncated.shape == (3,)
    finally:
        env.close()
