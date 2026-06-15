"""Vectorized DuelEnv construction and a throughput benchmark.

Training workers only ever import stickfighter.core/stickfighter.ai, never
pygame or pymunk, so there is nothing unpicklable to worry about across
multiprocessing workers. The one real constraint is Windows' 'spawn' start
method (the multiprocessing default here): every env factory passed to
AsyncVectorEnv must be picklable by plain stdlib pickle, which a lambda or a
closure over a nested function is NOT -- hence the top-level callable class
below instead of the more obvious `lambda: DuelEnv(...)`.
"""

import argparse
import time
