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


def make_run_dir(run_dir_arg: str | None) -> Path:
    run_dir = Path(run_dir_arg) if run_dir_arg else Path("runs") / time.strftime(
        "stickfighter_%Y%m%d_%H%M%S"
    )
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(exist_ok=True)
    (run_dir / "best").mkdir(exist_ok=True)
    (run_dir / "recordings").mkdir(exist_ok=True)
    return run_dir


def find_latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    # filenames are zero-padded (ckpt_{step:012d}.pt), so lexicographic sort
    # is numeric sort -- no risk of "ckpt_9.pt" sorting after "ckpt_100.pt".
    checkpoints = sorted(checkpoints_dir.glob("ckpt_*.pt"))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    obs_normalizer: RunningNormalizer,
    curriculum: Curriculum,
    trainee_elo: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "obs_mean": obs_normalizer.mean,
            "obs_var": obs_normalizer.var,
            "obs_count": obs_normalizer.count,
            "stage_idx": curriculum.stage_idx,
            "trainee_elo": trainee_elo,
        },
        path,
    )


def linear_schedule(start: float, end: float, progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return start + (end - start) * progress


def build_opponent(
    curriculum: Curriculum, league: League, model: ActorCritic, obs_normalizer, trainee_elo: float
) -> tuple[Policy, object | None]:
    """Returns (opponent, league_entry). league_entry is None outside S3."""
    if curriculum.stage.name != SELFPLAY_STAGE_NAME:
        return curriculum.stage.opponent_factory(), None
    if league.is_empty():
        league.add_snapshot(model, name="step_0", elo=trainee_elo)
    entry = league.sample_entry()
    opponent = league.make_opponent(entry, OBS_DIM, ACTION_DIM, obs_normalizer)
    return opponent, entry


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    set_seed(args.seed)
    lr_final = args.lr_final if args.lr_final is not None else args.lr * 0.1
