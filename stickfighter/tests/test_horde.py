from stickfighter.core.horde import TOTAL_ENEMIES, HordeSim


def _run_to_completion(sim: HordeSim, seed: int, max_steps: int = 20_000) -> int:
    sim.reset(seed=seed)
    steps = 0
    while not sim.cleared and sim.hero.alive and steps < max_steps:
        sim.step()
        steps += 1
    return steps


def test_hero_clears_the_full_horde_without_dying():
    for seed in range(5):
        sim = HordeSim()
        steps = _run_to_completion(sim, seed)
        assert sim.cleared, f"seed {seed} did not clear within {steps} steps"
        assert sim.hero.alive
        assert sim.total_kills == TOTAL_ENEMIES
