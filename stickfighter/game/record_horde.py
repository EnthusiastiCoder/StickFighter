"""Renders a full HordeSim run to a video file: the hero-vs-melee-waves
highlight scenario, not a playable mode. Same "cheap sim trains, pretty
renderer records" split as evaluate.py::record_match, reusing the existing
EffectsSystem/RagdollManager unchanged since HordeSim emits the same Event
types core/match.py does.
"""

import argparse
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio
import pygame

from stickfighter.core import constants as C
from stickfighter.core.horde import ARCHER_TOTAL, TOTAL_ENEMIES, HordeSim
from stickfighter.render import effects, ragdoll, theme
from stickfighter.render.draw import draw_arrow
from stickfighter.render.horde_draw import draw_cloud, draw_melee_enemy

TOTAL_TARGETS = TOTAL_ENEMIES + ARCHER_TOTAL

_ARCHER_DRAW = None  # lazy import to avoid a hard pygame dependency at module load for tests


def _draw_hero(surface, hero, elapsed, scale: float, victory: bool = False) -> None:
    global _ARCHER_DRAW
    if _ARCHER_DRAW is None:
        from stickfighter.render.draw import draw_archer

        _ARCHER_DRAW = draw_archer
    _ARCHER_DRAW(surface, hero, elapsed, victory=victory, scale=scale)


def _write_log(out_path: str, log: list[dict]) -> str:
    log_path = os.path.splitext(out_path)[0] + "_log.jsonl"
    with open(log_path, "w") as f:
        for entry in log:
            f.write(json.dumps(entry) + "\n")
    return log_path


def _draw_kill_counter(
    surface: pygame.Surface, kills: int, font: pygame.font.Font, scale: float
) -> None:
    text = font.render(f"KILLS: {kills}/{TOTAL_TARGETS}", True, theme.INK)
    surface.blit(text, (round(16 * scale), round(12 * scale)))


def _draw_archer(surface, cloud, elapsed, scale: float) -> None:
    global _ARCHER_DRAW
    if _ARCHER_DRAW is None:
        from stickfighter.render.draw import draw_archer

        _ARCHER_DRAW = draw_archer
    draw_cloud(surface, cloud.state.x, cloud.state.y, scale)
    _ARCHER_DRAW(surface, cloud.state, elapsed, scale=scale)


def _draw_ground(surface: pygame.Surface, scale: float) -> None:
    ground_rect = pygame.Rect(
        0, round(C.GROUND_Y * scale), round(C.ARENA_WIDTH * scale), round(40 * scale)
    )
    pygame.draw.rect(surface, theme.INK, ground_rect)


def record(
    out_path: str, seed: int = 0, fps: int = 60, max_steps: int = 20_000, scale: float = 3.0
) -> dict:
    """scale renders the whole scene at scale*960 x scale*540 -- genuinely
    more source pixels for every anti-aliased line/circle to work with, not
    just an upscaled blur of a small image. Every draw call in the pipeline
    (hero, enemies, arrows, particles, ragdolls) takes the same scale so
    everything lands in the same place relative to each other; only
    core/constants.py (read by the *simulation*, not just rendering) stays
    untouched, since scaling those would silently change hitboxes/ballistics
    instead of just how the frame looks.
    """
    pygame.init()
    font = pygame.font.Font(None, round(32 * scale))
    surface = pygame.Surface((round(C.ARENA_WIDTH * scale), round(C.ARENA_HEIGHT * scale)))

    sim = HordeSim()
    sim.reset(seed=seed)
    fx = effects.EffectsSystem()
    ragdolls = ragdoll.RagdollManager()
    elapsed = 0.0

    with imageio.get_writer(out_path, fps=fps) as writer:
        steps = 0
        while not sim.cleared and sim.hero.alive and steps < max_steps:
            events = sim.step()
            elapsed += C.DT
            steps += 1
            fx.handle_events(events)
            ragdolls.handle_events(events)
            fx.update(C.DT, sim.snapshot().arrows)
            ragdolls.update(C.DT)

            snap = sim.snapshot()
            surface.fill(theme.BG)
            _draw_ground(surface, scale)
            for cloud in snap.archers:
                _draw_archer(surface, cloud, elapsed, scale)
            for enemy in snap.enemies:
                draw_melee_enemy(surface, enemy, elapsed, scale=scale)
            _draw_hero(surface, snap.hero, elapsed, scale)
            for arrow in snap.arrows:
                draw_arrow(surface, arrow, scale=scale)
            fx.draw(surface, scale)
            ragdolls.draw(surface, scale)
            _draw_kill_counter(surface, snap.total_kills, font, scale)

            writer.append_data(pygame.surfarray.array3d(surface).swapaxes(0, 1))

        # hold on a victory pose so the video ends on a beat instead of
        # cutting off the instant the last enemy dies
        if sim.cleared:
            snap = sim.snapshot()
            for _ in range(int(fps * 1.5)):
                elapsed += C.DT
                surface.fill(theme.BG)
                _draw_ground(surface, scale)
                _draw_hero(surface, snap.hero, elapsed, scale, victory=True)
                _draw_kill_counter(surface, snap.total_kills, font, scale)
                writer.append_data(pygame.surfarray.array3d(surface).swapaxes(0, 1))

    log_path = _write_log(out_path, sim.log)
    return {
        "cleared": sim.cleared, "hero_alive": sim.hero.alive, "kills": sim.total_kills,
        "steps": steps, "log_path": log_path,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="horde.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--scale", type=float, default=3.0, help="render resolution multiplier")
    args = parser.parse_args(argv)

    result = record(args.out, seed=args.seed, fps=args.fps, scale=args.scale)
    print(f"wrote {args.out}: {result}")


if __name__ == "__main__":
    main()
