"""Slow integration test: PPO must show real, stable learning progress on
S0 (static target practice) within a bounded step budget. Not a "solves
reliably" bar (that needs far more than 100k steps for a 5D continuous
action space) -- a directly measurable floor that would fail if the
training loop were silently broken (wrong GAE, a reward sign flip, an
entropy collapse, exploding KL, etc).
