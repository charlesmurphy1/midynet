import multiprocessing as mp
import numpy as np
import time

from .statistics import MCStatistics
from .mcmc import logEvidence, logPosterior


class MultiProcess:
    def __init__(self, num_procs=1):
        self.num_procs = num_procs

    def func(self, inputs):
        raise NotImplementedError()

    def compute(self, inputs):
        with mp.Pool(self.num_procs) as p:
            out = p.map(self.func, inputs)
        return out


class Expectation(MultiProcess):
    def __init__(self, num_procs=1, seed=None):
        MultiProcess.__init__(self, num_procs=num_procs)
        self.seed = int(time.time()) if seed is None else seed

    def compute(self, num_samples=1):
        seeds = self.seed + np.arange(num_samples)
        with mp.Pool(self.num_procs) as p:
            out = p.map(self.func, seeds)
        return out
