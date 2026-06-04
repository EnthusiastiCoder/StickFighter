"""The renderer contract: reading a MatchSnapshot to draw a frame must never
mutate the sim state underneath it (render/ only ever reads)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

from stickfighter.core import constants as C  # noqa: E402
from stickfighter.core.match import MatchSim  # noqa: E402
from stickfighter.render import draw, effects, hud, ragdoll  # noqa: E402

pygame.init()


def _random_played_sim(seed: int, steps: int = 90) -> MatchSim:
    import random

    from stickfighter.core.actions import Action

    rng = random.Random(seed)
    sim = MatchSim()
    sim.reset(seed=seed)
    for _ in range(steps):
        if sim.match_over:
            break
        a0 = Action(
            move=rng.uniform(-1, 1),
            jump=rng.random() < 0.05,
            aim_angle=rng.uniform(-3.14, 3.14),
            draw=rng.random(),
            release=rng.random() < 0.1,
        )
        a1 = Action(
            move=rng.uniform(-1, 1),
            jump=rng.random() < 0.05,
            aim_angle=rng.uniform(-3.14, 3.14),
            draw=rng.random(),
            release=rng.random() < 0.1,
        )
        sim.step(a0, a1)
    return sim


def test_drawing_a_full_frame_does_not_mutate_sim_state():
    sim = _random_played_sim(seed=11)
    before = sim.state_tuple()

    snapshot = sim.snapshot()
    surface = pygame.Surface((int(C.ARENA_WIDTH), int(C.ARENA_HEIGHT)))
    fx = effects.EffectsSystem()
    ragdolls = ragdoll.RagdollManager()

    surface.fill((0, 0, 0))
    draw.draw_arena(surface, snapshot.arena)
    for archer in snapshot.archers:
        draw.draw_archer(surface, archer, elapsed=1.23)
    for arrow in snapshot.arrows:
        draw.draw_arrow(surface, arrow)
    fx.update(C.DT, snapshot.arrows)
    fx.draw(surface)
    ragdolls.update(C.DT)
    ragdolls.draw(surface)
    hud.draw_hud(surface, snapshot)

    after = sim.state_tuple()
    assert before == after


def test_effects_and_ragdolls_do_not_mutate_sim_state_across_event_batches():
    sim = _random_played_sim(seed=12)
    fx = effects.EffectsSystem()
    ragdolls = ragdoll.RagdollManager()

    events = sim.step(*_idle_actions())
    after_step = sim.state_tuple()
    fx.handle_events(events)
    ragdolls.handle_events(events)
    fx.update(C.DT, tuple(sim.arrows))
    ragdolls.update(C.DT)
