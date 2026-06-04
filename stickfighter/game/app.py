"""App entrypoint: window, fixed-timestep loop, and scene switching."""

import pygame

from stickfighter.core import constants as C
from stickfighter.game.scenes.menu import MenuScene

WINDOW_TITLE = "StickFighter"
MAX_FRAME_DT = 0.25  # clamp to avoid a spiral of death after a stall


def run() -> None:
    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    screen = pygame.display.set_mode((int(C.ARENA_WIDTH), int(C.ARENA_HEIGHT)))
    clock = pygame.time.Clock()

    scene = MenuScene()
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
            next_scene = scene.update(C.DT)
            if next_scene is not None:
                scene = next_scene
            accumulator -= C.DT

        scene.draw(screen)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run()
