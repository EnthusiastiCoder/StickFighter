"""PPO training CLI: curriculum-driven learning from scripted bots through a
self-play league of frozen past checkpoints.

Uses gymnasium's SyncVectorEnv exclusively. AsyncVectorEnv (real
multiprocessing workers, see envs/vec.py) was benchmarked against it and did
NOT outperform it at this project's scale -- the sim + bot computation is
cheap enough that multiprocessing IPC overhead ate the theoretical
parallelism gain. Running everything in one process also means a self-play
opponent's weights can just be swapped in place (`sub_env.opponent = ...`),
with no cross-process weight-sync mechanism needed at all.
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from stickfighter.ai.bots.base import Policy
from stickfighter.ai.bots.idle import IdleBot
from stickfighter.ai.envs.duel_env import DuelEnv
from stickfighter.ai.obs import OBS_DIM
from stickfighter.ai.ppo.curriculum import SELFPLAY_STAGE_NAME, Curriculum
from stickfighter.ai.ppo.league import DEFAULT_ELO, League
from stickfighter.ai.ppo.logging import TrainingLogger
from stickfighter.ai.ppo.policy import ActorCritic, RunningNormalizer
from stickfighter.ai.ppo.ppo import PPOConfig, RolloutBuffer, ppo_update
from stickfighter.ai.seeding import set_seed
from stickfighter.core.actions import ACTION_DIM


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=256, help="steps per env per rollout")
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="'cpu', 'cuda', or 'auto'")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-final", type=float, default=None, help="default: 10%% of --lr")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--ent-coef-final", type=float, default=0.001)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--n-minibatches", type=int, default=4)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--resume", default=None, help="a run directory to resume from")
    p.add_argument("--checkpoint-freq", type=int, default=20, help="rollouts between checkpoints")
    p.add_argument("--league-refresh-interval", type=int, default=20, help="rollouts")
    p.add_argument("--target-win-rate", type=float, default=None)
    p.add_argument("--target-streak", type=int, default=3)
    p.add_argument("--stage", type=int, default=None, help="force-start at this curriculum stage")
    return p.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
