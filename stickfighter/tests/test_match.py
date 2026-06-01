import random

from stickfighter.core import constants as C
from stickfighter.core.actions import Action
from stickfighter.core.arena import ARENAS
from stickfighter.core.ballistics import solve_launch_angle
from stickfighter.core.entities import head_hitbox, torso_hitbox
from stickfighter.core.events import ArrowHit, CoinEarned, Death, MatchOver, RoundOver
from stickfighter.core.match import MatchSim

NOOP = Action(move=0.0, jump=False, aim_angle=0.0, draw=0.0, release=False)
_FLAT = next(a for a in ARENAS if a.name == "flat")


def _random_action(rng: random.Random) -> Action:
    return Action(
        move=rng.uniform(-1, 1),
        jump=rng.random() < 0.05,
        aim_angle=rng.uniform(-3.14159, 3.14159),
        draw=rng.random(),
        release=rng.random() < 0.1,
    )


def _aim_action(shooter, target_point: tuple[float, float]) -> Action:
    dx = target_point[0] - shooter.x
    dy = target_point[1] - (shooter.top() + C.BOW_HEIGHT_OFFSET)
    angle = solve_launch_angle(dx, dy, C.ARROW_SPEED_MAX, C.GRAVITY, prefer_low=True)
    assert angle is not None
    return Action(move=0.0, jump=False, aim_angle=angle, draw=1.0, release=True)


def _fire_and_collect_until_landed(sim: MatchSim, target_point: tuple[float, float], max_steps=120):
    """Fires one aimed shot from archer 0, then keeps stepping (both archers
    idle) until the arrow lands -- it takes many frames to cross the arena,
    so a single step() only spawns it. Returns every event along the way."""
    events = list(sim.step(_aim_action(sim.archers[0], target_point), NOOP))
    for _ in range(max_steps):
        if any(isinstance(e, (ArrowHit, RoundOver)) for e in events):
            break
        events.extend(sim.step(NOOP, NOOP))
    return events


def test_same_seed_and_action_tape_is_bit_identical():
    rng = random.Random(99)
    tape = [(_random_action(rng), _random_action(rng)) for _ in range(2000)]

    sim_a = MatchSim()
    sim_a.reset(seed=99)
    for a0, a1 in tape:
        sim_a.step(a0, a1)

    sim_b = MatchSim()
    sim_b.reset(seed=99)
    for a0, a1 in tape:
        sim_b.step(a0, a1)

    assert sim_a.state_tuple() == sim_b.state_tuple()


def test_headshot_is_an_instakill_regardless_of_max_hp():
    sim = MatchSim()
    sim.reset(seed=1, arena=_FLAT)
    target = head_hitbox(sim.archers[1])
    events = _fire_and_collect_until_landed(sim, (target.cx, target.cy))

    hits = [e for e in events if isinstance(e, ArrowHit)]
    deaths = [e for e in events if isinstance(e, Death)]
    assert len(hits) == 1 and hits[0].part == "head"
    assert hits[0].damage >= C.CHAR_MAX_HP  # instakill regardless of remaining HP
    assert len(deaths) == 1
    assert deaths[0] == Death(who=1, killer=0, headshot=True)
    # the kill ends the round (and this match isn't over yet, so a new one
    # started immediately -- both archers are back to full HP for it)
    assert sim.p0_rounds == 1
    assert sim.archers[0].kills == 1
    assert sim.archers[1].alive and sim.archers[1].hp == C.CHAR_MAX_HP


def test_torso_hit_applies_partial_damage_and_no_death():
    sim = MatchSim()
    sim.reset(seed=2, arena=_FLAT)
    target = torso_hitbox(sim.archers[1])
    mid = (target.left + target.right) / 2, (target.top + target.bottom) / 2
    events = _fire_and_collect_until_landed(sim, mid)

    hits = [e for e in events if isinstance(e, ArrowHit)]
    assert len(hits) == 1 and hits[0].part == "torso"
    assert sim.archers[1].hp == C.CHAR_MAX_HP - C.DAMAGE_TORSO
    assert sim.archers[1].alive
    assert not any(isinstance(e, Death) for e in events)


def test_hit_awards_coins_to_the_shooter():
    sim = MatchSim()
    sim.reset(seed=3, arena=_FLAT)
    target = head_hitbox(sim.archers[1])
    events = _fire_and_collect_until_landed(sim, (target.cx, target.cy))

    coin_events = [e for e in events if isinstance(e, CoinEarned)]
    assert coin_events == [CoinEarned(who=0, amount=C.COINS_HEADSHOT)]
    assert sim.archers[0].coins == C.COINS_HEADSHOT


def test_round_timeout_with_equal_hp_is_a_draw_and_match_continues():
    sim = MatchSim()
    sim.reset(seed=4, arena=_FLAT)
    round_overs = []
    for _ in range(1900):
        events = sim.step(NOOP, NOOP)
        round_overs.extend(e for e in events if isinstance(e, RoundOver))
        if round_overs:
            break

    assert len(round_overs) == 1
    assert round_overs[0].winner is None
    assert round_overs[0].p0_rounds == 0
    assert round_overs[0].p1_rounds == 0
    assert not sim.match_over
    # a fresh round started: both archers are back to full HP and alive
    assert sim.archers[0].hp == C.CHAR_MAX_HP and sim.archers[0].alive
    assert sim.archers[1].hp == C.CHAR_MAX_HP and sim.archers[1].alive


def test_match_ends_after_win_rounds_reached_and_then_step_is_a_noop():
    sim = MatchSim()
    sim.reset(seed=5, arena=_FLAT)
