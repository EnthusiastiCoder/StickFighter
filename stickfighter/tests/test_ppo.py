import pytest
import torch

from stickfighter.ai.obs import OBS_DIM
from stickfighter.ai.ppo.policy import ActorCritic
from stickfighter.ai.ppo.ppo import PPOConfig, RolloutBuffer, ppo_update
from stickfighter.core.actions import ACTION_DIM

GAMMA = 0.9
LAMBDA = 0.8


def _buffer_from(rewards, values, dones, obs_dim=1, action_dim=1) -> RolloutBuffer:
    device = torch.device("cpu")
    buf = RolloutBuffer(
        len(rewards), n_envs=1, obs_dim=obs_dim, action_dim=action_dim, device=device
    )
    for r, v, d in zip(rewards, values, dones, strict=True):
        buf.add(
            torch.zeros(1, obs_dim),
            torch.zeros(1, action_dim),
            torch.zeros(1),
            [r],
            [d],
            torch.tensor([v]),
        )
    return buf


def test_gae_matches_hand_computed_values_with_no_episode_boundary():
    # gamma=0.9, lambda=0.8, rewards=[1,2,3], values=[0.5,0.6,0.7], last_value=0.8
    # worked by hand back-to-front: A_2=3.02, A_1=4.2044, A_0=4.067168
    buf = _buffer_from(rewards=[1.0, 2.0, 3.0], values=[0.5, 0.6, 0.7], dones=[0.0, 0.0, 0.0])
    buf.compute_gae(torch.tensor([0.8]), gamma=GAMMA, gae_lambda=LAMBDA)

    expected_advantages = [4.067168, 4.2044, 3.02]
    actual_advantages = buf.advantages.squeeze(-1).tolist()
    for actual, expected in zip(actual_advantages, expected_advantages, strict=True):
        assert actual == pytest.approx(expected, abs=1e-4)

    expected_returns = [a + v for a, v in zip(expected_advantages, [0.5, 0.6, 0.7], strict=True)]
    actual_returns = buf.returns.squeeze(-1).tolist()
    for actual, expected in zip(actual_returns, expected_returns, strict=True):
        assert actual == pytest.approx(expected, abs=1e-4)


def test_gae_does_not_bootstrap_across_an_episode_boundary():
    # same rewards/values, but the episode ends exactly at t=1: the
    # advantage at t=1 must not include any of A_2's future return.
    buf = _buffer_from(rewards=[1.0, 2.0, 3.0], values=[0.5, 0.6, 0.7], dones=[0.0, 1.0, 0.0])
    buf.compute_gae(torch.tensor([0.8]), gamma=GAMMA, gae_lambda=LAMBDA)

    expected_advantages = [2.048, 1.4, 3.02]
    actual_advantages = buf.advantages.squeeze(-1).tolist()
    for actual, expected in zip(actual_advantages, expected_advantages, strict=True):
        assert actual == pytest.approx(expected, abs=1e-4)


def test_get_batches_covers_every_sample_exactly_once_per_epoch():
    buf = _buffer_from(
        rewards=[0.0] * 8, values=[0.0] * 8, dones=[0.0] * 8, obs_dim=2, action_dim=2
    )
    buf.compute_gae(torch.tensor([0.0]), gamma=0.99, gae_lambda=0.95)
    seen = 0
    for batch in buf.get_batches(n_minibatches=4):
        obs_b = batch[0]
        seen += obs_b.shape[0]
    assert seen == buf.n_steps * buf.n_envs


def test_ppo_update_changes_parameters_and_returns_finite_stats():
    torch.manual_seed(0)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    config = PPOConfig(n_epochs=2, n_minibatches=2)

    n_steps, n_envs = 16, 4
    buf = RolloutBuffer(n_steps, n_envs, OBS_DIM, ACTION_DIM, torch.device("cpu"))
    obs = torch.randn(n_envs, OBS_DIM)
    for _ in range(n_steps):
        with torch.no_grad():
            action, log_prob, _entropy, value = model.get_action_and_value(obs)
        reward = torch.randn(n_envs).numpy()
        done = (torch.rand(n_envs) < 0.05).float().numpy()
        buf.add(obs, action, log_prob, reward, done, value)
        obs = torch.randn(n_envs, OBS_DIM)

    with torch.no_grad():
        last_value = model.get_value(obs)
    buf.compute_gae(last_value, config.gamma, config.gae_lambda)
