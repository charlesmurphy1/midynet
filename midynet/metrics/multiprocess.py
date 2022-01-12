import multiprocessing as mp
import numpy as np
import time
from dataclasses import dataclass, field

from .statistics import MCStatistics

__all__ = ["MultiProcess", "Expectation"]


@dataclass
class MultiProcess:
    num_procs: int = 1

    def func(self, inputs):
        raise NotImplementedError()

    def compute(self, inputs):
        with mp.Pool(self.num_procs) as p:
            out = p.map(self.func, inputs)
        return out


@dataclass
class Expectation(MultiProcess):
    seed: int = field(default_factory=lambda: int(time.time()))
    # statistics: str = field(default_factory=lambda: MCStatistics("std"))

    def func(self, seed: int) -> float:
        raise NotImplementedError()

    def compute(self, num_samples: int = 1) -> list[float]:
        seeds = self.seed + np.arange(num_samples)
        with mp.Pool(self.num_procs) as p:
            out = p.map(self.func, seeds)
        return out
