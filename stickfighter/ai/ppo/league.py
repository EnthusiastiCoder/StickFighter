"""Self-play opponent pool: frozen policy snapshots sampled via PFSP
(prioritized fictitious self-play -- weight opponents the trainee is
currently losing to more heavily), plus Elo ratings to track skill
progression across the league.
"""

import copy
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from stickfighter.ai.obs import encode_obs
from stickfighter.ai.ppo.policy import ActorCritic
from stickfighter.core.actions import Action
from stickfighter.core.match import MatchSnapshot

DEFAULT_ELO = 1200.0
ELO_K = 24.0


@dataclass
class LeagueEntry:
    name: str
    state_dict: dict
    elo: float = DEFAULT_ELO
    games: int = 0
    wins_vs_trainee: int = 0


@dataclass
class FrozenPolicyBot:
    """Wraps a frozen ActorCritic snapshot as a Policy for match play."""

    model: ActorCritic
    obs_normalizer: object | None = field(default=None)

    def __post_init__(self) -> None:
        self.model.eval()

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, snapshot: MatchSnapshot, perspective: int) -> Action:
        obs = encode_obs(snapshot, perspective)
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = self.model.get_action_and_value(obs_t)
        return Action.from_array(action.squeeze(0).numpy())


class League:
    def __init__(
        self, pfsp_power: float = 2.0, max_size: int = 30, seed: int | None = None
    ) -> None:
        self.entries: list[LeagueEntry] = []
        self.pfsp_power = pfsp_power
        self.max_size = max_size
        self._rng = random.Random(seed)

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def add_snapshot(self, model: ActorCritic, name: str, elo: float | None = None) -> LeagueEntry:
        state_dict = copy.deepcopy(model.state_dict())
        seed_elo = elo if elo is not None else self._average_elo()
        entry = LeagueEntry(name=name, state_dict=state_dict, elo=seed_elo)
        self.entries.append(entry)
        if len(self.entries) > self.max_size:
            self.entries.sort(key=lambda e: e.elo)
            self.entries.pop(0)  # drop the weakest snapshot to bound memory/compute
        return entry

    def _average_elo(self) -> float:
        if not self.entries:
            return DEFAULT_ELO
        return sum(e.elo for e in self.entries) / len(self.entries)

    def sample_entry(self) -> LeagueEntry:
        if not self.entries:
            raise ValueError("cannot sample from an empty league")
        # win_rate is the ENTRY's win rate against the trainee -- high means
        # the trainee is currently losing to it, which is exactly the
        # opponent PFSP should serve up more of.
        win_rates = [
            (e.wins_vs_trainee / e.games) if e.games > 0 else 0.5 for e in self.entries
        ]
        weights = [wr**self.pfsp_power + 1e-3 for wr in win_rates]
        return self._rng.choices(self.entries, weights=weights, k=1)[0]

    def make_opponent(
        self, entry: LeagueEntry, obs_dim: int, action_dim: int, obs_normalizer=None
    ) -> FrozenPolicyBot:
        model = ActorCritic(obs_dim, action_dim)
        model.load_state_dict(entry.state_dict)
        return FrozenPolicyBot(model, obs_normalizer)

    def record_match(self, entry: LeagueEntry, trainee_elo: float, entry_won: bool) -> float:
        """Updates PFSP bookkeeping and both Elo ratings; returns the
        trainee's new Elo."""
        entry.games += 1
        if entry_won:
            entry.wins_vs_trainee += 1

        expected_entry = 1.0 / (1.0 + 10 ** ((trainee_elo - entry.elo) / 400.0))
        score_entry = 1.0 if entry_won else 0.0
        entry.elo += ELO_K * (score_entry - expected_entry)
        return trainee_elo + ELO_K * ((1.0 - score_entry) - (1.0 - expected_entry))
