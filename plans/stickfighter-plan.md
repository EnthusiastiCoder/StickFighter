# StickFighter — Stickman Archery Duel + Trainable AI

## Context

The repo is empty (only tooling/config). The goal is to build a faithful clone of the
*Archers 2*–style stickman archery duel from the reference screenshot — mustard-yellow
background, black stick-figure silhouettes on floating platforms, arced arrows with white
trails, ragdoll deaths, headshot bonuses, coin + kill HUD — **and**, as the primary motive,
to train an AI to actually be *good* at it.

"Train the AI" is the constraint that shapes the whole architecture: the game logic must be a
**pure, deterministic, headless-runnable simulation** with no rendering dependency, so
thousands of matches per second can be played in parallel. Everything else (pygame renderer,
menus, shop) is a consumer of that core, never a participant in it.

**Confirmed choices:** Python + pygame · pymunk for ragdolls · real-time duel with free
movement · scripted ballistic bot first, then PPO self-play. All remaining design decisions
were delegated to me and are recorded below as decisions, not options.

---

## Environment (decided)

Use the **global Python 3.12** (`C:\Python312`, invoked as `py -3.12`). It is the shared
interpreter every project venv was seeded from (`home = C:\Python312`) and already contains
everything expensive:

| Present | Version |
|---|---|
| torch (CUDA working) | 2.6.0+cu124, `cuda.is_available() == True` |
| gymnasium | 1.3.0 |
| pymunk | 7.2.0 |
| numpy / scipy / matplotlib | 2.4.6 / 1.17.1 / 3.10.3 |
| numba, tensorboardX, imageio, tqdm, rich | ✓ |
| pytest, pytest-cov, ruff | ✓ |

I checked every venv under `Projects\`: none has the full combo (`Machine Learning\Simulations`
has pygame+pymunk but no torch; `Voice Moduler\.venv` has torch but no pygame/pymunk). The
global env is the only one that is one package away from complete.

**Only install: `pygame-ce`** (maintained drop-in fork; it is what your frozen requirements
already used). One package, used by no other project → negligible blast radius on the shared env.

**Deliberately NOT installing `stable-baselines3`** — `sb3[extra]` would drag opencv/pandas and
could bump `gymnasium`/`numpy` inside a *shared* interpreter, and SB3's VecEnv API fights
self-play leagues. We write our own PPO (~350 lines, full control over hybrid actions + league).

Hardware notes that constrain defaults: 8 cores, RTX 4050 **6 GB**. Your Mario project recorded
`--n-envs 11` blowing up OpenBLAS on this exact box → **default `--n-envs 8`**. The policy is a
tiny MLP, so **default `--device cpu`** (GPU offload for 2×256 MLPs is usually a net loss);
`--device auto` available and benchmarked.

---

## Architecture

The single most important rule: **`core/` never imports pygame, torch, or gymnasium.**

```
stickfighter/
  core/                  # PURE SIM — deterministic, headless, no rendering deps
    constants.py         # dt=1/60, gravity, speeds, damage table, arena 960x540
    events.py            # ArrowFired, ArrowHit, Damage, Death, RoundOver, CoinEarned
    actions.py           # Action dataclass  <->  np.float32[5]
    entities.py          # ArcherState, ArrowState, hitbox layout (head/torso/arms/legs)
    arena.py             # Arena (platform AABBs, ground, spawns) + 8 layouts + random_arena(rng)
    character.py         # kinematic platformer controller (accel, jump, AABB resolution)
    ballistics.py        # arrow integration + swept-segment collision + solve_launch_angle()
    rules.py             # damage/HP/headshot, coins, round & match state machine, timeout
    match.py             # MatchSim.reset(seed) / .step(a0, a1) -> list[Event] / .snapshot()
  render/                # pygame; READS snapshots, never mutates sim
    theme.py draw.py effects.py hud.py ragdoll.py
  game/                  # human-facing app
    input.py app.py scenes/{menu,duel,practice,spectate,shop,results,settings}.py
  ai/
    obs.py reward.py seeding.py
    bots/{base,ballistic,idle,wanderer}.py
    envs/{duel_env,vec}.py
    ppo/{policy,ppo,league,curriculum,logging,train,evaluate}.py
  tests/
plans/    research/    runs/ (gitignored)
```

### Why pymunk is render-only
