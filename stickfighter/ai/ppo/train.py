"""PPO training CLI: curriculum-driven learning from scripted bots through a
self-play league of frozen past checkpoints.

Uses gymnasium's SyncVectorEnv exclusively. AsyncVectorEnv (real
multiprocessing workers, see envs/vec.py) was benchmarked against it and did
NOT outperform it at this project's scale -- the sim + bot computation is
cheap enough that multiprocessing IPC overhead ate the theoretical
parallelism gain. Running everything in one process also means a self-play
opponent's weights can just be swapped in place (`sub_env.opponent = ...`),
with no cross-process weight-sync mechanism needed at all.
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
