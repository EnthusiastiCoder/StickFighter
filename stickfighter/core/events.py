"""Events emitted by MatchSim.step(). Consumed by the renderer, reward function,
and logging -- never by the sim itself, so they can be frozen/immutable."""

from dataclasses import dataclass

Part = str  # one of "head", "torso", "limb"
