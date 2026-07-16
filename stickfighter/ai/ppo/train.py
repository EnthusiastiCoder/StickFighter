"""PPO training CLI: curriculum-driven learning from scripted bots through a
self-play league of frozen past checkpoints.

Uses gymnasium's SyncVectorEnv exclusively. AsyncVectorEnv (real
multiprocessing workers, see envs/vec.py) is not just unhelpful here but
actively dangerous on a machine with a CUDA build of torch installed: every
worker process independently loads the full cuBLAS/cuFFT DLL stack (never
even used, since training runs on CPU), and spawning more than a couple of
such processes has been observed to exhaust this machine's paging file
(WinError 1455) badly enough to crash unrelated processes system-wide.
Running everything in one process sidesteps that entirely, and also means a
self-play opponent's weights can just be swapped in place
(`sub_env.opponent = ...`), with no cross-process weight-sync needed.

On CPU, the right thread count is different for the two halves of a
rollout, benchmarked here on this machine: collecting the rollout does one
tiny inference call (batch = n_envs) per env-step, and multi-threaded BLAS's
own synchronization overhead exceeds the payoff at that size -- 1 thread
measured faster than 8 at every n_envs tried. ppo_update instead runs full
epochs over the whole rollout in much larger minibatches, where more
threads measured ~3x faster (1084ms -> 367ms at 1 vs 8 threads on this
machine). So thread count is switched around the update call rather than
fixed for the whole run.
"""

import argparse
import os
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
    p.add_argument("--n-envs", type=int, default=64)
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
) -> tuple[Policy, object | None, float]:
    """Returns (opponent, league_entry, near_miss_scale). league_entry is
    None outside S3. near_miss_scale matches whichever stage's opponent was
    actually picked -- a rehearsal match against an easier stage's opponent
    must get that easier stage's shaping bonus, not the current stage's
    (usually zeroed-out) one, or rehearsal loses most of its point."""
    if curriculum.stage.name != SELFPLAY_STAGE_NAME:
        stage = curriculum.opponent_stage()
        return stage.opponent_factory(), None, stage.near_miss_scale
    if league.is_empty():
        league.add_snapshot(model, name="step_0", elo=trainee_elo)
    entry = league.sample_entry()
    opponent = league.make_opponent(entry, OBS_DIM, ACTION_DIM, obs_normalizer)
    return opponent, entry, curriculum.stage.near_miss_scale


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    rollout_threads, update_threads = 1, (os.cpu_count() or 8)
    if device.type == "cpu":
        torch.set_num_threads(rollout_threads)
    set_seed(args.seed)
    lr_final = args.lr_final if args.lr_final is not None else args.lr * 0.1

    run_dir = make_run_dir(args.run_dir)
    logger = TrainingLogger(run_dir / "logs")

    model = ActorCritic(OBS_DIM, ACTION_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    obs_normalizer = RunningNormalizer((OBS_DIM,))
    curriculum = Curriculum()
    if args.stage is not None:
        curriculum.stage_idx = args.stage
    league = League(seed=args.seed)
    global_step = 0
    trainee_elo = DEFAULT_ELO

    if args.resume:
        ckpt_path = find_latest_checkpoint(Path(args.resume) / "checkpoints")
        if ckpt_path is None:
            raise FileNotFoundError(f"no checkpoints found under {args.resume}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt["global_step"]
        obs_normalizer.mean = ckpt["obs_mean"]
        obs_normalizer.var = ckpt["obs_var"]
        obs_normalizer.count = ckpt["obs_count"]
        curriculum.stage_idx = ckpt["stage_idx"]
        trainee_elo = ckpt["trainee_elo"]
        print(f"resumed from {ckpt_path} at step {global_step}, stage {curriculum.stage.name}")

    envs = [DuelEnv(opponent=IdleBot(), near_miss_scale=curriculum.stage.near_miss_scale)
            for _ in range(args.n_envs)]
    vec_env = gym.vector.SyncVectorEnv([(lambda e=e: e) for e in envs])
    league_entries: list[object | None] = [None] * args.n_envs

    def refresh_opponents() -> None:
        for i, sub_env in enumerate(envs):
            opponent, entry, near_miss_scale = build_opponent(
                curriculum, league, model, obs_normalizer, trainee_elo
            )
            sub_env.opponent = opponent
            sub_env.near_miss_scale = near_miss_scale
            league_entries[i] = entry

    refresh_opponents()
    obs, _ = vec_env.reset(seed=args.seed)

    ppo_config = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        n_epochs=args.n_epochs,
        n_minibatches=args.n_minibatches,
    )
    buffer = RolloutBuffer(args.n_steps, args.n_envs, OBS_DIM, ACTION_DIM, device)
    rollout_idx = 0
    target_hits = 0

    while global_step < args.total_steps:
        buffer.reset()
        for _ in range(args.n_steps):
            obs_normalizer.update(obs)
            normed_obs = obs_normalizer.normalize(obs)
            obs_t = torch.as_tensor(normed_obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, log_prob, _entropy, value = model.get_action_and_value(obs_t)
            next_obs, reward, terminated, truncated, _infos = vec_env.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)
            buffer.add(obs_t, action, log_prob, reward, done.astype(np.float32), value)
            global_step += args.n_envs

            advanced = False
            for i in range(args.n_envs):
                if not terminated[i]:
                    continue
                trainee_won = envs[i].sim.winner == 0
                advanced = curriculum.record_match(trainee_won) or advanced
                if league_entries[i] is not None:
                    trainee_elo = league.record_match(
                        league_entries[i], trainee_elo, entry_won=not trainee_won
                    )
            if advanced:
                refresh_opponents()

            obs = next_obs

        with torch.no_grad():
            last_obs_t = torch.as_tensor(
                obs_normalizer.normalize(obs), dtype=torch.float32, device=device
            )
            last_value = model.get_value(last_obs_t)
        buffer.compute_gae(last_value, ppo_config.gamma, ppo_config.gae_lambda)

        progress = global_step / args.total_steps
        for group in optimizer.param_groups:
            group["lr"] = linear_schedule(args.lr, lr_final, progress)
        ppo_config.ent_coef = linear_schedule(args.ent_coef, args.ent_coef_final, progress)

        if device.type == "cpu":
            torch.set_num_threads(update_threads)
        stats = ppo_update(model, optimizer, buffer, ppo_config)
        if device.type == "cpu":
            torch.set_num_threads(rollout_threads)
        rollout_idx += 1

        logger.log(
            global_step,
            {
                "curriculum/stage": curriculum.stage_idx,
                "curriculum/win_rate": curriculum.win_rate,
                "league/trainee_elo": trainee_elo,
                "league/size": len(league.entries),
                "opt/lr": optimizer.param_groups[0]["lr"],
                "opt/ent_coef": ppo_config.ent_coef,
                **{f"ppo/{k}": v for k, v in stats.items()},
            },
        )

        if curriculum.stage.name == SELFPLAY_STAGE_NAME:
            if rollout_idx % args.league_refresh_interval == 0:
                league.add_snapshot(model, name=f"step_{global_step}", elo=trainee_elo)
                refresh_opponents()

        if rollout_idx % args.checkpoint_freq == 0:
            ckpt_path = run_dir / "checkpoints" / f"ckpt_{global_step:012d}.pt"
            save_checkpoint(
                ckpt_path, model, optimizer, global_step, obs_normalizer, curriculum, trainee_elo
            )

        if args.target_win_rate is not None and len(curriculum.results) >= curriculum.window:
            if curriculum.win_rate >= args.target_win_rate:
                target_hits += 1
            else:
                target_hits = 0
            if target_hits >= args.target_streak:
                print(
                    f"target win rate {args.target_win_rate} sustained for "
                    f"{args.target_streak} rollouts -- stopping early at step {global_step}"
                )
                break

    save_checkpoint(
        run_dir / "final.pt", model, optimizer, global_step, obs_normalizer, curriculum, trainee_elo
    )
    logger.close()
    print(f"training complete: {global_step} steps, stage {curriculum.stage.name}")
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
