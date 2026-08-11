# Wolf and dragon 3D models (real animated poses)

The traced-vector-outline approach used for the hound and dragon (see
`spider-and-harpy-silhouette-references.md` for the tracing technique)
still read as "one flat image being moved around" even after cutting
individual pieces (legs, wings) and rotating them by formula — the user
asked explicitly for real 2D or 3D **models**, not hand-computed motion.
This note documents the models actually used to fix that.

## Source

**Poly Pizza** (https://poly.pizza), not Sketchfab — Sketchfab requires a
logged-in account to download even CC0 models (confirmed by hitting its
login wall), and a plain automated browser session gets blocked by Google's
sign-in security check (`navigator.webdriver === true`) regardless of which
real browser/profile is behind it. Poly Pizza's models are downloadable
without any login.

- **Wolf**: https://poly.pizza/m/P1gU3Qkr9r — Quaternius, **CC0** (public
  domain). Animations: `Walk`, `Gallop`, `Idle`, `Idle_2`, `Eating`, `Attack`,
  `Death`, `Jump_ToIdle`, `Idle_HitReact_Left/Right` (also duplicated with an
  `AnimalArmature|` prefix — both names work, use the un-prefixed one).
  `Walk` duration: 1.0417s.
- **Dragon**: https://poly.pizza/m/VBvzjFIYws — Quaternius, **CC0**.
  Animations (all prefixed `DragonArmature|`): `Dragon_Attack`,
  `Dragon_Attack2`, `Dragon_Death`, `Dragon_Flying`, `Dragon_Hit`.
  `Dragon_Flying` duration: 1.6667s.

## Getting the actual GLB without the UI
