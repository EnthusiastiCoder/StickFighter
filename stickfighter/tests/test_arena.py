from stickfighter.core import constants as C
from stickfighter.core.arena import ARENAS, random_arena
import random


def _body_range(x: float, y: float) -> tuple[float, float, float, float]:
    return (x - C.CHAR_HALF_WIDTH, x + C.CHAR_HALF_WIDTH, y - C.CHAR_HEIGHT, y)


def test_all_arenas_within_bounds():
    for arena in ARENAS:
        for p in arena.platforms:
            assert p.x >= 0.0
            assert p.right <= C.ARENA_WIDTH
            assert 0.0 <= p.y <= C.ARENA_HEIGHT


def test_spawns_rest_on_ground_and_avoid_floating_platforms():
    for arena in ARENAS:
        for x, y in (arena.spawn_p0, arena.spawn_p1):
            assert y == C.GROUND_Y
            bl, br, bt, bb = _body_range(x, y)
            for platform in arena.platforms[1:]:  # skip ground itself
                horizontal_overlap = platform.x < br and platform.right > bl
                vertical_overlap = bt < platform.y < bb
                assert not (horizontal_overlap and vertical_overlap), (
                    f"{arena.name}: spawn at ({x},{y}) clips platform {platform}"
                )


def test_spawns_are_on_opposite_sides():
    for arena in ARENAS:
        assert arena.spawn_p0[0] < arena.spawn_p1[0]


def test_random_arena_is_deterministic_for_a_seeded_rng():
    a = random_arena(random.Random(123))
    b = random_arena(random.Random(123))
    assert a.name == b.name


def test_random_arena_only_returns_known_layouts():
    rng = random.Random(0)
    names = {a.name for a in ARENAS}
    for _ in range(50):
        assert random_arena(rng).name in names
