import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

pygame.init()
pygame.display.set_mode((1, 1))

from stickfighter.game.difficulty import DIFFICULTIES  # noqa: E402
from stickfighter.game.scenes.duel import DuelScene  # noqa: E402
from stickfighter.game.watch_ai import parse_args  # noqa: E402


def test_parse_args_defaults_to_legend_vs_gold():
    args = parse_args([])
    assert args.left == "Legend"
    assert args.right == "Gold"
    assert args.seed is None


def test_parse_args_accepts_any_difficulty_pair_and_a_seed():
    args = parse_args(["--left", "Bronze", "--right", "Silver", "--seed", "3"])
    assert args.left == "Bronze"
    assert args.right == "Silver"
    assert args.seed == 3


def test_parsed_args_build_a_working_bot_vs_bot_duel_scene():
    args = parse_args(["--left", "Bronze", "--right", "Silver", "--seed", "3"])
    scene = DuelScene(
        opponent_factory=DIFFICULTIES[args.right],
        player0_factory=DIFFICULTIES[args.left],
        seed=args.seed,
    )
    assert scene.player0 is not None  # bot-controlled, not human
    scene.update(1 / 60)
