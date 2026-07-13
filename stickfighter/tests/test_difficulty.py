import torch

from stickfighter.ai.bots.ballistic import BallisticBot
from stickfighter.ai.obs import OBS_DIM
from stickfighter.ai.ppo.curriculum import Curriculum
from stickfighter.ai.ppo.evaluate import TrainedPolicyBot
from stickfighter.ai.ppo.policy import ActorCritic, RunningNormalizer
from stickfighter.ai.ppo.train import save_checkpoint
from stickfighter.game import difficulty


def test_all_difficulties_produce_a_working_policy_without_a_trained_checkpoint(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    for factory in difficulty.DIFFICULTIES.values():
        assert factory() is not None
    # with no runs/ dir at all, Legend must gracefully fall back to a scripted bot
    assert isinstance(difficulty.make_legend(), BallisticBot)


def test_find_trained_checkpoint_returns_none_with_no_runs_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert difficulty.find_trained_checkpoint() is None
