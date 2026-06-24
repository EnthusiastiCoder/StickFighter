"""Custom PPO: rollout buffer with GAE, clipped surrogate objective, clipped
value loss, advantage normalization, gradient clipping. Written from scratch
rather than stable-baselines3 so it can drive a self-play league without
fighting someone else's VecEnv/policy abstractions.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from stickfighter.ai.ppo.policy import ActorCritic


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    n_epochs: int = 4
    n_minibatches: int = 4
    normalize_advantages: bool = True


class RolloutBuffer:
    def __init__(
        self, n_steps: int, n_envs: int, obs_dim: int, action_dim: int, device: torch.device
    ) -> None:
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.obs = torch.zeros((n_steps, n_envs, obs_dim), device=device)
        self.actions = torch.zeros((n_steps, n_envs, action_dim), device=device)
        self.log_probs = torch.zeros((n_steps, n_envs), device=device)
        self.rewards = torch.zeros((n_steps, n_envs), device=device)
        self.dones = torch.zeros((n_steps, n_envs), device=device)
        self.values = torch.zeros((n_steps, n_envs), device=device)
        self.advantages = torch.zeros((n_steps, n_envs), device=device)
        self.returns = torch.zeros((n_steps, n_envs), device=device)
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: np.ndarray,
        done: np.ndarray,
        value: torch.Tensor,
    ) -> None:
        """`done[i]` means the transition stored at index i itself ended the
        episode -- i.e. whatever obs a training loop puts at index i+1 (if
        any) belongs to a fresh episode and must not be bootstrapped across."""
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
        self.dones[i] = torch.as_tensor(done, device=self.device, dtype=torch.float32)
        self.values[i] = value
        self.ptr += 1

    def reset(self) -> None:
        self.ptr = 0

    def compute_gae(self, last_value: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        """`last_value` bootstraps past the buffer's final step -- the value
        estimate of whatever observation follows it (irrelevant, and never
        used, if that final step was itself terminal: dones[-1] zeroes it).
        """
        last_gae = torch.zeros(self.n_envs, device=self.device)
        for t in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get_batches(self, n_minibatches: int):
        batch_size = self.n_steps * self.n_envs
        flat_obs = self.obs.reshape(batch_size, -1)
        flat_actions = self.actions.reshape(batch_size, -1)
        flat_log_probs = self.log_probs.reshape(batch_size)
        flat_advantages = self.advantages.reshape(batch_size)
        flat_returns = self.returns.reshape(batch_size)
        flat_values = self.values.reshape(batch_size)

        indices = torch.randperm(batch_size, device=self.device)
        minibatch_size = batch_size // n_minibatches
        for start in range(0, batch_size, minibatch_size):
            idx = indices[start : start + minibatch_size]
            yield (
                flat_obs[idx],
                flat_actions[idx],
                flat_log_probs[idx],
                flat_advantages[idx],
                flat_returns[idx],
                flat_values[idx],
            )


def ppo_update(
    model: ActorCritic, optimizer: torch.optim.Optimizer, buffer: RolloutBuffer, config: PPOConfig
) -> dict[str, float]:
    stats: dict[str, list[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": [],
    }
    for _ in range(config.n_epochs):
        for obs_b, act_b, old_log_prob_b, adv_b, ret_b, old_value_b in buffer.get_batches(
            config.n_minibatches
        ):
            if config.normalize_advantages:
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
