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
