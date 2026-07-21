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
