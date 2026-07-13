import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

pygame.init()

from stickfighter.ai.bots.ballistic import PERFECT, BallisticBot  # noqa: E402
from stickfighter.ai.bots.idle import IdleBot  # noqa: E402
from stickfighter.core.events import CoinEarned  # noqa: E402
from stickfighter.game import save  # noqa: E402
from stickfighter.game.scenes.duel import DuelScene  # noqa: E402


def test_duel_scene_defaults_to_a_human_controlled_player0():
    scene = DuelScene(opponent_factory=IdleBot)
    assert scene.player0 is None
    assert scene.input is not None


def test_duel_scene_supports_bot_vs_bot_spectate_mode():
    scene = DuelScene(opponent_factory=IdleBot, player0_factory=lambda: BallisticBot(PERFECT))
    assert scene.player0 is not None
    assert scene.input is None
    for _ in range(30):
        scene.update(1 / 60)  # must not touch the (deliberately null) self.input


def test_accumulate_coins_persists_only_archer0s_coins(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    scene = DuelScene(opponent_factory=IdleBot)
    assert scene.save_data.total_coins == 0

    scene._accumulate_coins([CoinEarned(who=0, amount=3), CoinEarned(who=1, amount=5)])
    assert scene.save_data.total_coins == 3
    assert save.load().total_coins == 3

    scene._accumulate_coins([CoinEarned(who=0, amount=1)])
    assert scene.save_data.total_coins == 4
    assert save.load().total_coins == 4


def test_accumulate_coins_ignores_events_for_the_other_archer(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    scene = DuelScene(opponent_factory=IdleBot)
    scene._accumulate_coins([])
    scene._accumulate_coins([CoinEarned(who=1, amount=10)])
    assert scene.save_data.total_coins == 0
    assert not (tmp_path / "save.json").exists()  # never wrote when nothing changed


def test_spectate_mode_never_persists_coins(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    scene = DuelScene(
        opponent_factory=IdleBot, player0_factory=lambda: BallisticBot(PERFECT), seed=0
    )
    for _ in range(300):
        if scene.sim.match_over:
            break
        scene.update(1 / 60)
    assert scene.save_data.total_coins == 0
    assert save.load().total_coins == 0
