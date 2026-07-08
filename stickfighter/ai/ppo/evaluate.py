"""Evaluate a trained checkpoint: win rates against the scripted bot ladder,
and an optional recorded match GIF/MP4 using the real pygame renderer -- the
"cheap sim trains, pretty renderer records" split, driven by the exact same
MatchSim used everywhere else in this project.
"""

import argparse

import torch

from stickfighter.ai.bots.ballistic import EASY, HARD, MEDIUM, PERFECT, BallisticBot
from stickfighter.ai.bots.idle import IdleBot
from stickfighter.ai.obs import OBS_DIM, encode_obs
from stickfighter.ai.ppo.policy import ActorCritic, RunningNormalizer
from stickfighter.core.actions import ACTION_DIM, Action
from stickfighter.core.match import MatchSim

BOT_LADDER = {
    "idle": IdleBot,
    "easy": lambda: BallisticBot(EASY),
    "medium": lambda: BallisticBot(MEDIUM),
    "hard": lambda: BallisticBot(HARD),
    "perfect": lambda: BallisticBot(PERFECT),
}


class TrainedPolicyBot:
    def __init__(
        self, model: ActorCritic, normalizer: RunningNormalizer, deterministic: bool = True
    ):
        self.model = model
        self.normalizer = normalizer
        self.deterministic = deterministic

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, snapshot, perspective: int) -> Action:
        obs = self.normalizer.normalize(encode_obs(snapshot, perspective))
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if self.deterministic:
                action = self.model.actor(obs_t)
            else:
                action, _, _, _ = self.model.get_action_and_value(obs_t)
        return Action.from_array(action.squeeze(0).numpy())


def load_checkpoint(path) -> tuple[ActorCritic, RunningNormalizer, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ActorCritic(OBS_DIM, ACTION_DIM)
    model.load_state_dict(ckpt["model"])
    model.eval()
    normalizer = RunningNormalizer((OBS_DIM,))
    normalizer.mean = ckpt["obs_mean"]
    normalizer.var = ckpt["obs_var"]
    normalizer.count = ckpt["obs_count"]
    return model, normalizer, ckpt


def play_match(policy_a, policy_b, seed: int, max_steps: int = 20_000) -> int | None:
    sim = MatchSim()
    sim.reset(seed=seed)
    policy_a.reset(seed=seed)
    policy_b.reset(seed=seed)
    for _ in range(max_steps):
        if sim.match_over:
            break
        sim.step(policy_a.act(sim.snapshot(), 0), policy_b.act(sim.snapshot(), 1))
    return sim.winner


def evaluate_win_rates(
    model: ActorCritic, normalizer: RunningNormalizer, episodes: int, seed: int = 0
) -> dict[str, float]:
    results = {}
    for name, factory in BOT_LADDER.items():
        wins = 0
        for i in range(episodes):
            trained = TrainedPolicyBot(model, normalizer)
            winner = play_match(trained, factory(), seed=seed + i)
            wins += winner == 0
        results[name] = wins / episodes
    return results


def record_match(
    model: ActorCritic,
    normalizer: RunningNormalizer,
    opponent_name: str,
    out_path: str,
    seed: int = 0,
    max_steps: int = 1800,
    fps: int = 60,
) -> int | None:
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import imageio
    import pygame

    from stickfighter.core import constants as C
    from stickfighter.render import draw, effects, hud, ragdoll, theme

    pygame.init()
    surface = pygame.Surface((int(C.ARENA_WIDTH), int(C.ARENA_HEIGHT)))

    sim = MatchSim()
    sim.reset(seed=seed)
    trained = TrainedPolicyBot(model, normalizer)
    opponent = BOT_LADDER[opponent_name]()
    trained.reset(seed=seed)
    opponent.reset(seed=seed)

    fx = effects.EffectsSystem()
    ragdolls = ragdoll.RagdollManager()
    elapsed = 0.0

    # streamed straight to disk -- buffering the whole match as a frame list
    # first can run into gigabytes for a long/undecided match.
    with imageio.get_writer(out_path, fps=fps) as writer:
        for _ in range(max_steps):
            if sim.match_over:
                break
            events = sim.step(trained.act(sim.snapshot(), 0), opponent.act(sim.snapshot(), 1))
            elapsed += C.DT
            fx.handle_events(events)
            ragdolls.handle_events(events)
            fx.update(C.DT, tuple(sim.arrows))
            ragdolls.update(C.DT)

            snapshot = sim.snapshot()
            surface.fill(theme.BG)
            draw.draw_arena(surface, snapshot.arena)
            for archer in snapshot.archers:
                draw.draw_archer(surface, archer, elapsed)
            for arrow in snapshot.arrows:
                draw.draw_arrow(surface, arrow)
            fx.draw(surface)
            ragdolls.draw(surface)
            hud.draw_hud(surface, snapshot)

            writer.append_data(pygame.surfarray.array3d(surface).swapaxes(0, 1))
