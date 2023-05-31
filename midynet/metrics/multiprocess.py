import multiprocessing as mp
import time
import numpy as np

from typing import Optional
from multiprocessing import get_context

__all__ = ("MultiProcess", "Expectation")


# class NoDaemonProcess(mp.Process):
#     @property
#     def daemon(self):
#         return False

#     @daemon.setter
#     def daemon(self, value):
#         pass


# class NoDaemonContext(type(mp.get_context())):
#     Process = NoDaemonProcess


# class NestablePool(multiprocessing.pool.Pool):
#     def __init__(self, *args, **kwargs):
#         kwargs["context"] = NoDaemonContext()
#         super(NestablePool, self).__init__(*args, **kwargs)


class MultiProcess:
    def __init__(self, inputs):
        self.inputs = inputs

    def func(self, inputs):
        raise NotImplementedError()

    def compute(self, pool: Optional[mp.Pool] = None):
        if pool is None:
            return [self.func(x) for x in self.inputs]
        out = pool.map(self.func, self.inputs)
        return out

    def compute_async(self, pool: Optional[mp.Pool] = None):
        if pool is None:
            return self.compute()
        return pool.map_async(self.func, self.inputs)


class Expectation(MultiProcess):
    def __init__(self, seed=None, n_samples=1):
        self.seed = seed if seed is not None else int(time.time())
        super().__init__(self.seed + np.arange(n_samples))
