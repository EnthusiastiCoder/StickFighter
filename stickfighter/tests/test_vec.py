"""Tests exercise SyncVectorEnv, which runs the exact same _EnvFactory /
DuelEnv construction logic as AsyncVectorEnv, just in-process. AsyncVectorEnv
itself (real multiprocessing workers) was verified manually via a standalone
script under a proper `if __name__ == "__main__":` guard -- the way it will
actually be invoked from ppo/train.py -- since spawning real subprocesses
from inside a pytest run is a separate, flakier concern from the env logic
this test suite is responsible for.
"""
