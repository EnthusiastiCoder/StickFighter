# StickFighter

A stickman archery duel game (in the vein of Archers 2) with a trainable AI opponent.

Two archers on floating platforms draw and fire arrows at each other -- headshots kill
instantly, body shots chip health, movement and timing matter. The core simulation is a
pure, deterministic, headless module with no rendering dependency, so it doubles as a
`gymnasium`-compatible training environment: the same code that runs the playable game
also runs thousands of self-play matches per second to train a PPO agent.

## Project layout

- `stickfighter/core/` -- pure simulation: physics, rules, match state machine. No pygame,
  no torch, no gymnasium.
- `stickfighter/render/` -- pygame rendering of a core snapshot. Never mutates sim state.
- `stickfighter/game/` -- the playable app: scenes, input, menus.
- `stickfighter/ai/` -- bots, the Gymnasium env, and the PPO self-play training pipeline.
- `stickfighter/tests/` -- pytest suite.

See `plans/` for the build plan and `research/` for design notes.

## Setup

This project reuses the shared global Python 3.12 interpreter (`C:\Python312`) rather than
a project-local virtualenv, since that environment already has `torch` (CUDA), `gymnasium`,
`pymunk`, and the rest of the ML stack installed. See `requirements.txt` for exact versions.

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m pytest -q
```

## Running things

```powershell
py -3.12 -m stickfighter.game.app                          # play the game
py -3.12 -m stickfighter.ai.ppo.train --stage 0             # train the AI
py -3.12 -m stickfighter.ai.ppo.evaluate --ckpt <path>       # watch a trained checkpoint play
```
