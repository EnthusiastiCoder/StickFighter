"""HordeSim: a hardcoded hero archer vs. waves of hardcoded melee enemies.

Not a game mode and not RL -- a standalone deterministic scenario built to
render one highlight video. The hero is fully autonomous (PERFECT-tier
ballistic aim, no reaction delay), so there is no external Action input at
all: reset(seed) + repeated step() is the whole interface.
