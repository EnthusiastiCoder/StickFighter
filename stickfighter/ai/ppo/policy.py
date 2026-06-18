"""Actor-critic network: separate policy and value MLPs (2x256, tanh),
state-independent log_std, orthogonal init -- the standard modern
continuous-control PPO recipe (as in CleanRL's ppo_continuous_action.py),
adapted with a final Tanh on the policy mean since our action space is
bounded to [-1, 1] rather than unbounded.
"""
