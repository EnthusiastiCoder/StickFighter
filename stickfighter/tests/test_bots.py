import math

from stickfighter.ai.bots.ballistic import HARD, BallisticBot
from stickfighter.ai.bots.idle import IdleBot
from stickfighter.ai.bots.wanderer import WandererBot
from stickfighter.core.actions import NOOP
from stickfighter.core.arena import ARENAS
from stickfighter.core.match import MatchSim

_FLAT = next(a for a in ARENAS if a.name == "flat")


def _play_match(seed: int, bot0, bot1, arena=None, max_steps: int = 20_000) -> int | None:
    sim = MatchSim()
    sim.reset(seed=seed, arena=arena)
    bot0.reset()
    bot1.reset()
    for _ in range(max_steps):
        if sim.match_over:
            break
        sim.step(bot0.act(sim.snapshot(), 0), bot1.act(sim.snapshot(), 1))
    return sim.winner


def test_idle_bot_always_returns_noop():
    sim = MatchSim()
    sim.reset(seed=0)
    bot = IdleBot()
    assert bot.act(sim.snapshot(), 0) == NOOP
    assert bot.act(sim.snapshot(), 1) == NOOP


def test_wanderer_bot_varies_its_actions_over_time():
    sim = MatchSim()
    sim.reset(seed=0)
    bot = WandererBot(seed=1)
    actions = [bot.act(sim.snapshot(), 0) for _ in range(300)]
    assert len({a.aim_angle for a in actions}) > 10
    assert any(a.release for a in actions)
