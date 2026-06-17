"""Vectorized DuelEnv construction and a throughput benchmark.

Training workers only ever import stickfighter.core/stickfighter.ai, never
pygame or pymunk, so there is nothing unpicklable to worry about across
multiprocessing workers. The one real constraint is Windows' 'spawn' start
method (the multiprocessing default here): every env factory passed to
AsyncVectorEnv must be picklable by plain stdlib pickle, which a lambda or a
closure over a nested function is NOT -- hence the top-level callable class
below instead of the more obvious `lambda: DuelEnv(...)`.
"""

import argparse
import time

import gymnasium as gym

from stickfighter.ai.bots.ballistic import MEDIUM, BallisticBot
from stickfighter.ai.bots.base import Policy
from stickfighter.ai.envs.duel_env import DuelEnv
from stickfighter.core.arena import Arena


def default_opponent_factory() -> Policy:
    return BallisticBot(MEDIUM)


class _EnvFactory:
    """A picklable env constructor: multiprocessing pickles this object (an
    int and a top-level function reference, both fine) and calls it inside
    the worker process to build that worker's own DuelEnv instance."""

    def __init__(self, worker_index: int, opponent_factory, arena: Arena | None, **env_kwargs):
        self.worker_index = worker_index
        self.opponent_factory = opponent_factory
        self.arena = arena
        self.env_kwargs = env_kwargs

    def __call__(self) -> DuelEnv:
        return DuelEnv(opponent=self.opponent_factory(), arena=self.arena, **self.env_kwargs)


def make_vec_env(
    num_envs: int,
    opponent_factory=default_opponent_factory,
    arena: Arena | None = None,
    asynchronous: bool = True,
    **env_kwargs,
) -> gym.vector.VectorEnv:
    env_fns = [_EnvFactory(i, opponent_factory, arena, **env_kwargs) for i in range(num_envs)]
    cls = gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv
    return cls(env_fns)


def benchmark(worker_counts: list[int], steps_per_worker: int = 500) -> None:
    for n in worker_counts:
        env = make_vec_env(n, asynchronous=(n > 1))
        env.reset(seed=0)
        t0 = time.perf_counter()
        for _ in range(steps_per_worker):
            actions = env.action_space.sample()
            env.step(actions)
        elapsed = time.perf_counter() - t0
        total_steps = n * steps_per_worker
        print(
            f"n_envs={n:>3}  {total_steps / elapsed:>9.0f} steps/sec  "
            f"({elapsed:.2f}s for {total_steps} steps)"
        )
        env.close()
