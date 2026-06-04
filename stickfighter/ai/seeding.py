"""Central seeding so every source of randomness (Python's random, NumPy,
and PyTorch) is reproducible from one seed, with per-worker offsets so
vectorized envs do not all sample identical random streams."""

import random

import numpy as np

_WORKER_STRIDE = 100_003  # arbitrary large odd-ish stride, keeps worker streams apart


def set_seed(seed: int, worker_index: int = 0) -> int:
    effective = seed + worker_index * _WORKER_STRIDE
    random.seed(effective)
    np.random.seed(effective % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(effective)
    except ImportError:
        pass
    return effective
