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
