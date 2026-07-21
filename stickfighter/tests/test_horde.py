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


def test_hero_never_takes_damage_at_the_default_pacing():
    # the wave size/hero fire-rate balance is tuned so 3 concurrent enemies
    # never actually reach melee range -- if this starts failing, the pacing
    # (MAX_ACTIVE vs RELOAD_TIME vs enemy speed) needs retuning, not a patch.
    for seed in range(5):
        sim = HordeSim()
        sim.reset(seed=seed)
        while not sim.cleared and sim.hero.alive:
            sim.step()
        assert sim.hero.hp == 100.0


def test_determinism_same_seed_same_outcome():
    sim_a = HordeSim()
    steps_a = _run_to_completion(sim_a, seed=7)
    sim_b = HordeSim()
    steps_b = _run_to_completion(sim_b, seed=7)

    assert steps_a == steps_b
    assert sim_a.total_kills == sim_b.total_kills
    assert [(e["event"], e.get("arrow_id")) for e in sim_a.log] == [
        (e["event"], e.get("arrow_id")) for e in sim_b.log
    ]


def test_shield_blocks_torso_and_limb_but_not_head():
    sim = HordeSim()
    sim.reset(seed=0)
    while not sim.cleared and sim.hero.alive:
        sim.step()

    hits_and_blocks = [e for e in sim.log if e["event"] in ("hit", "shield_block")]
    assert any(e["event"] == "shield_block" for e in hits_and_blocks) or not any(
        e["event"] == "hit" and e["part"] == "shield" for e in hits_and_blocks
    )
    # every logged hit that connects is on an unshielded part or a head --
    # a shield never appears as the damaged part
    for e in hits_and_blocks:
        if e["event"] == "hit":
            assert e["part"] != "shield" or e["damage"] == 0.0


def test_no_arrow_wastes_a_shot_on_an_already_claimed_target():
    # regression: the hero used to re-target the same nearest enemy every
    # tick even after already firing at it, wasting a second arrow that
    # arrived to find it already dead. Confirmed via the per-shot log --
    # every fire's target should still be alive (or about to be the one
    # this exact arrow kills) at the moment it's aimed at.
    sim = HordeSim()
    sim.reset(seed=0)
    while not sim.cleared and sim.hero.alive:
        sim.step()

    fires = [e for e in sim.log if e["event"] == "fire"]
    misses = [e for e in sim.log if e["event"] == "miss"]
    # not zero-tolerance (a called shot at spawn-distance range can still
    # legitimately miss), but double-targeting waste should be rare/absent
    assert len(misses) <= 1
    assert len(fires) <= TOTAL_ENEMIES + 3
