# StickFighter

A stickman archery duel game (in the vein of Archers 2) with a trainable AI opponent.

Two archers on floating platforms draw and fire arrows at each other -- headshots kill
instantly, body shots chip health, movement and timing matter. The core simulation is a
pure, deterministic, headless module with no rendering dependency, so it doubles as a
`gymnasium`-compatible training environment: the same code that runs the playable game
also runs thousands of self-play matches per second to train a PPO agent.

![Horde highlight: the hero archer holds off melee waves and cloud archers](docs/media/horde_demo.gif)

_An 8-second excerpt of the horde highlight scenario (rendered with `record_horde`)._

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

## Playing

```powershell
py -3.12 -m stickfighter.game.app
```

Title screen -> pick a difficulty (Bronze/Silver/Gold are scripted bots of increasing
skill; Legend is whatever checkpoint training has produced so far, falling back to a
perfect-play scripted bot if nothing has been trained yet) or Spectate to watch two AIs
fight. Drag from your archer to aim and set draw strength; release to fire. WASD/arrows
move and jump. Esc pauses, Enter starts a rematch after a match ends, G toggles gore.
Coins earned in a real (non-spectate) duel persist to `save.json`.

## Horde highlight scenario

A standalone, deterministic scenario built to render a highlight video: an autonomous hero archer
in the centre holds off waves of melee enemies and archers on clouds closing in from both sides.
It is not a playable mode; it reuses the core ballistics, rules and effects unchanged.

```powershell
py -3.12 -m stickfighter.game.record_horde --out horde.mp4 --seed 612 --fps 60 --scale 3.0
```

## Training the AI

```powershell
py -3.12 -m stickfighter.ai.ppo.train --run-dir runs/my_run --n-envs 8
py -3.12 -m stickfighter.ai.ppo.train --resume runs/my_run          # continue later
py -3.12 -m stickfighter.ai.ppo.evaluate --ckpt runs/my_run/final.pt --episodes 100 \
    --record hard --record-out match.gif
```

Training runs a curriculum -- S0 static target practice, S1 a noisy scripted bot, S2 a
sharp scripted bot, S3 self-play against a league of frozen past checkpoints -- advancing
each stage automatically once a rolling win rate threshold is met. `evaluate.py` reports
win rate against the full scripted bot ladder and can record a match through the real
renderer as a GIF (or MP4, if `imageio-ffmpeg` is installed).

Point the game's Legend difficulty at a specific run by placing its `final.pt` (or
`best.pt`) at `runs/stickfighter_main/`, or it falls back to the newest checkpoint found
anywhere under `runs/`.

## Copyright

© Arpan Mandal. All Rights Reserved.

See [`COPYRIGHT`](./COPYRIGHT) for the terms.
