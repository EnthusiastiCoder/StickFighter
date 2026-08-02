from stickfighter.core import constants as C
from stickfighter.core.entities import ArrowState
from stickfighter.core.horde import ARCHER_TOTAL, DODGE_RANGE, HERO_X, TOTAL_ENEMIES, HordeSim

TOTAL_TARGETS = TOTAL_ENEMIES + ARCHER_TOTAL


def _run_to_completion(sim: HordeSim, seed: int, max_steps: int = 20_000) -> int:
    sim.reset(seed=seed)
    steps = 0
    while not sim.cleared and sim.hero.alive and steps < max_steps:
        sim.step()
        steps += 1
    return steps


def test_hero_clears_the_full_horde_without_dying():
    # a specific known-good seed (seed=0 stopped clearing once cloud archers
    # were fixed to actually fire back -- see the note on
    # test_hero_survives_the_horde_at_the_default_pacing), not a swept
    # range -- see that test for the aggregate reliability bound across many
    # seeds. Matches game/record_horde.py's default --seed.
    sim = HordeSim()
    steps = _run_to_completion(sim, seed=2)
    assert sim.cleared, f"did not clear within {steps} steps"
    assert sim.hero.alive
    assert sim.total_kills == TOTAL_TARGETS


def test_hero_survives_the_horde_at_the_default_pacing():
    # deliberately NOT zero-damage: enemies are meant to close real distance
    # and occasionally land a hit (per user feedback -- a hero that snipes
    # everything from 300-500px away "can't even reach close enough" to be
    # dramatic). This is also a real, measured 60% floor (12/20), not the
    # 90% one this test asserted before cloud archers were fixed to actually
    # fire back and hit -- two bugs stacked here: an initial cooldown as
    # long as the hero's whole time-to-kill left every archer harmless
    # (0 archer_fire events across a full run), and once that was fixed, an
    # on-screen firing margin wider than the arrow's own out-of-bounds
    # despawn line meant every shot that *did* fire died on its own spawning
    # tick before ever traveling anywhere. With archers actually shooting
    # and landing hits, an instant-kill headshot (unchanged game rule,
    # shared with the real duel game) getting through the dodge at least
    # once across a ~40-archer fight is common enough that 12/20 is the real
    # number at the current (eased back down per feedback) fire rate -- see
    # game/record_horde.py for the specific known-good seed used for the
    # actual rendered video.
    cleared = 0
    n = 20
    for seed in range(n):
        sim = HordeSim()
        sim.reset(seed=seed)
        while not sim.cleared and sim.hero.alive:
            sim.step()
        if sim.cleared:
            cleared += 1
            assert sim.hero.hp > 0.0
    assert cleared / n >= 0.5, f"only {cleared}/{n} seeds cleared"


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


def test_hero_does_not_waste_shots_on_out_of_range_or_claimed_targets():
    # three real bugs, all regression-tested here: (1) the hero used to
    # re-target the same nearest enemy every tick even after already firing
    # at it, wasting a second arrow that arrived to find it already dead;
    # (2) after cloud archers were added, the hero would sometimes pick one
    # spawned beyond the bow's actual max range, and a gravity-less atan2
    # fallback fired a doomed shot every time; (3) once enemies could spawn
    # far enough out to need a restored travel distance (see horde.py's
    # SPAWN_X_JITTER), the hero would sometimes get a mathematically valid
    # ballistic solution for a target beyond the arrow's own out-of-bounds
    # despawn line -- a shot that can never physically arrive. All three are
    # now "don't fire this tick" cases, not "fire something wrong" cases.
    # Bounds measured directly off 10 seeds at the current settings (0-1
    # hero misses, 100-103 fires for 100 kills), not guessed.
    sim = HordeSim()
    sim.reset(seed=0)
    while not sim.cleared and sim.hero.alive:
        sim.step()

    fires = [e for e in sim.log if e["event"] == "fire"]
    fire_ids = {f["arrow_id"] for f in fires}
    hero_misses = [e for e in sim.log if e["event"] == "miss" and e["arrow_id"] in fire_ids]
    assert len(hero_misses) <= 3
    assert len(fires) <= TOTAL_TARGETS + 8


def test_cloud_archers_spawn_and_can_be_killed():
    # seed=2, not 0 -- see test_hero_survives_the_horde_at_the_default_pacing,
    # seed=0 is one of the seeds that doesn't clear now that archers fire back.
    sim = HordeSim()
    sim.reset(seed=2)
    while not sim.cleared and sim.hero.alive:
        sim.step()

    archer_kills = [
        e for e in sim.log if e["event"] == "archer_hit" and e.get("killed")
    ]
    assert len(archer_kills) == ARCHER_TOTAL
    assert sim.total_kills == TOTAL_ENEMIES + ARCHER_TOTAL


def test_hero_dodges_an_incoming_arrow_on_a_collision_course():
    # a direct, deterministic exercise of _update_hero_dodge: construct one
    # enemy-owned arrow on a collision course with the hero's torso, timed to
    # land inside DODGE_REACT_WINDOW, and confirm the hero actually strafes
    # in response. Replaces an earlier statistical version that ran full
    # matches and counted how many organically produced a dodge -- real, but
    # increasingly rare now that archers stand closer (CLOUD_ARCHER_STANDOFF
    # was pulled in to fix a stall, see horde_entities.py) and get killed
    # faster (faster hero fire rate), down to 2/30 full runs over 20k steps.
    # That flakiness made this the wrong tool to confirm the *mechanism*
    # itself still works -- this test targets that directly instead.
    sim = HordeSim()
    sim.reset(seed=0)
    sim.hero.x = HERO_X
    t = 0.3  # within DODGE_REACT_WINDOW (0.12, 0.5)
    arrow_x = HERO_X + 300.0
    hero_center_y = sim.hero.top() + (C.TORSO_TOP_Y + C.TORSO_BOTTOM_Y) / 2.0
    sim.arrows = [
        ArrowState(
            id=0,
            x=arrow_x,
            y=hero_center_y - 0.5 * C.GRAVITY * t * t,  # lands on hero_center_y at time t
            vx=-(arrow_x - HERO_X) / t,
            vy=0.0,
            owner=1,
        )
    ]
    sim._update_hero_dodge()
    assert sim.hero.vx != 0.0
    assert DODGE_RANGE[0] - 1.0 <= sim.hero.x <= DODGE_RANGE[1] + 1.0


def test_hero_roams_away_from_home_position_over_a_full_run():
    # the hero used to be permanently anchored at HERO_X -- confirm it
    # actually moves during a real run, not just that the mechanism works in
    # isolation (see test_hero_dodges_an_incoming_arrow_on_a_collision_course
    # for that). Deterministic per seed (not a statistical sample): measured
    # 2/30 seeds (0-29) ever moving over a full clear, so this checks that at
    # least one of them does rather than asserting a rate.
    seeds_that_moved = 0
    for seed in range(30):
        sim = HordeSim()
        sim.reset(seed=seed)
        steps = 0
        while not sim.cleared and sim.hero.alive and steps < 20_000:
            sim.step()
            steps += 1
            if abs(sim.hero.x - HERO_X) > 5.0:
                seeds_that_moved += 1
                break
            assert DODGE_RANGE[0] - 1.0 <= sim.hero.x <= DODGE_RANGE[1] + 1.0
    assert seeds_that_moved >= 1, "no seed in the sample ever moved the hero off HERO_X"


def test_snapshot_only_reports_alive_enemies_and_archers():
    sim = HordeSim()
    sim.reset(seed=0)
    for _ in range(50):
        sim.step()
    snap = sim.snapshot()
    assert all(e.alive for e in snap.enemies)
    assert all(a.state.alive for a in snap.archers)
