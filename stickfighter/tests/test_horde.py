from stickfighter.core.horde import TOTAL_ENEMIES, HordeSim


def _run_to_completion(sim: HordeSim, seed: int, max_steps: int = 20_000) -> int:
    sim.reset(seed=seed)
    steps = 0
    while not sim.cleared and sim.hero.alive and steps < max_steps:
        sim.step()
        steps += 1
    return steps
