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


def test_find_trained_checkpoint_finds_the_latest_numeric_checkpoint(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ckpt_dir = tmp_path / "runs" / "some_run" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ckpt_000000000100.pt").write_bytes(b"")
    (ckpt_dir / "ckpt_000000000050.pt").write_bytes(b"")

    found = difficulty.find_trained_checkpoint()
    assert found.name == "ckpt_000000000100.pt"


def test_find_trained_checkpoint_prefers_final_over_raw_checkpoints(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "stickfighter_main"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "ckpt_000000000100.pt").write_bytes(b"")
    (run_dir / "final.pt").write_bytes(b"")
