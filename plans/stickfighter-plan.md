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
