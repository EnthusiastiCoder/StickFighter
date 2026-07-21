# Horde Mode — hardcoded hero vs. melee horde, for a highlight video

## Goal

Not a game mode for players, not RL — a standalone deterministic scenario built
purely to render one video: a hardcoded, PERFECT-tier archer (the hero, fixed on
the left) guns down waves of hardcoded melee enemies (swordsman / spearman /
shieldbearer) walking in from the right. Output is an MP4.

No neural net anywhere in this feature. Reuses the existing pure-sim primitives
(arrow ballistics, hitboxes, damage table) but is otherwise a new, self-contained
module tree so it cannot regress the 1v1 duel game or training pipeline.

## Confirmed decisions (from user)

- Multiple enemies on screen simultaneously (not one-at-a-time gauntlet).
- Enemies are melee (sword/spear), not archers — no bow, no ranged threat back.
  Some carry shields. Melee damage is high (a real, if rare, threat to the hero).
- Hero uses the existing `PERFECT` ballistic-bot tier: 0 aim noise, 0 reaction
  delay, full draw, headshot-capable.
- Enemy count/pacing/exact numbers: mine to decide (below).

## Design

**Arena**: flat ground only (`GROUND_Y`), no platforms — hero is stationary, so
no character-vs-platform physics is needed for this mode at all.

**Hero**: fixed position near the left edge, never moves. Each tick: if reload
ready and any enemy alive, target the *nearest* alive enemy; aim for the head
if it has a raised shield (bypasses the block), otherwise the torso (bigger
target, same 0-miss `solve_launch_angle` math — this is flavor, not accuracy,
since the hero never actually misses). Reuses `solve_launch_angle`,
`spawn_arrow`, `step_arrow`, `segment_hits_circle/rect` unchanged.

**Enemy types** (hardcoded presets, same spirit as `BallisticBotConfig`):

| type | hp | speed (px/s) | melee range | melee dmg | shield |
|---|---|---|---|---|---|
| swordsman | 30 | 150 | 45px | 25 | no |
| spearman | 30 | 130 | 65px (reach) | 30 | no |
| shieldbearer | 30 | 90 | 45px | 45 | yes (blocks torso/limb, not head) |

HP=30 means any unblocked hit (head/torso/limb, all ≥ `DAMAGE_LIMB`=16... torso
34/head 9999 both exceed 30, limb 16 does not — so limb needs 2 hits) — the
hero always goes for head-or-torso, both one-shot, so in practice every enemy
dies in exactly one arrow, except a shield-blocked torso shot bounces and does
nothing (a visible beat: shot bounces off shield → hero's next shot, still
tracking nearest-enemy each tick, naturally reassesses and can go head instead
since `has_shield` is already known — no separate "recognize the shield" state
needed, the per-tick target-part rule already conditions on it).

**Waves**: 15 enemies total, mix 50% swordsman / 30% spearman / 20%
shieldbearer (seeded RNG). Spawn off the right edge, staggered so at most 3 are
"active" (spawned) at once; when active count drops below 3 and the queue
isn't empty, spawn the next one. Keeps 2-3 visible/approaching at all times
without ever truly swarming the hero (hero's fire rate, `RELOAD_TIME`=0.45s ⇒
~2.2 shots/sec, comfortably clears 3 approaching enemies well before they
close the ~800px gap at 90-150px/s).

**Hero HP/death**: kept as a real mechanic (not scripted-invincible) as a
safety net, but the pacing above is chosen so the hero should not take a hit in
practice — if it ever does, that's a real signal the wave pacing needs
retuning, not a bug to hide.

## New files (all additive, nothing existing touched except a small draw.py
reuse of `_draw_legs`)

1. `stickfighter/core/horde_entities.py` — `MeleeEnemyState` dataclass +
   `EnemyConfig` presets (SWORDSMAN/SPEARMAN/SHIELDBEARER).
2. `stickfighter/core/horde_events.py` — `EnemySpawned`, `EnemyHit`,
   `ShieldBlocked`, `EnemyKilled`, `HeroHit`, `HeroDown`, `HordeCleared`.
3. `stickfighter/core/horde.py` — `HordeSim`: `reset(seed)`, `step()`,
   `snapshot()`. Pure, headless, deterministic (no pygame/torch imports),
   consistent with the rest of `core/`.
4. `stickfighter/render/horde_draw.py` — `draw_melee_enemy()` (reuses
   `draw._draw_legs` for the walk cycle; new head/torso/weapon/shield drawing).
5. `stickfighter/game/record_horde.py` — CLI: runs `HordeSim` headlessly,
   renders each frame to an off-screen pygame surface (`SDL_VIDEODRIVER=dummy`,
   same pattern as `evaluate.py::record_match`), writes an MP4 via imageio.
6. `stickfighter/tests/test_horde.py` — determinism (same seed ⇒ identical
   state tape), shield blocks torso/limb but not head, a full 15-enemy horde
   is clearable within a bounded step budget, hero HP never reaches 0 at the
   default config.
