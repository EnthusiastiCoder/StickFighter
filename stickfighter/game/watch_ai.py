"""Standalone launcher: opens a real window straight into AI-vs-AI spectate
mode, skipping the title/mode-select screens, for quickly watching a
specific matchup of trained/scripted difficulties play live.

    py -3.12 -m stickfighter.game.watch_ai --left Legend --right Gold
"""

import argparse

import pygame

from stickfighter.core import constants as C
from stickfighter.game.difficulty import DIFFICULTIES
from stickfighter.game.scenes.duel import DuelScene

MAX_FRAME_DT = 0.25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", choices=list(DIFFICULTIES), default="Legend")
    parser.add_argument("--right", choices=list(DIFFICULTIES), default="Gold")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pygame.init()
    pygame.display.set_caption(f"StickFighter -- {args.left} vs {args.right}")
    screen = pygame.display.set_mode((int(C.ARENA_WIDTH), int(C.ARENA_HEIGHT)))
    clock = pygame.time.Clock()

    scene = DuelScene(
        opponent_factory=DIFFICULTIES[args.right],
        player0_factory=DIFFICULTIES[args.left],
        seed=args.seed,
    )
    accumulator = 0.0
    running = True
    while running:
        frame_dt = min(clock.tick(240) / 1000.0, MAX_FRAME_DT)
        accumulator += frame_dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                scene.handle_event(event)

        while accumulator >= C.DT:
            scene.update(C.DT)
            accumulator -= C.DT

        scene.draw(screen)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run()
