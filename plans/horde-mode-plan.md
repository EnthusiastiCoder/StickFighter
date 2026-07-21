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
