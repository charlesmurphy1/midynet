import numpy as np
import time
from dataclasses import dataclass, field

from midynet.config import *
from _midynet import utility
from .multiprocess import MultiProcess, Expectation
from .statistics import Statistics
from .metrics import Metrics

__all__ = ["DynamicsPredictionEntropy", "DynamicsPredictionEntropyMetrics"]


@dataclass
class DynamicsPredictionEntropy(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        utility.seed(seed)
        dynamics = DynamicsFactory.build(self.config.dynamics)
        graph = RandomGraphFactory.build(self.config.graph)
        dynamics.set_random_graph(graph.get_wrap())
        dynamics.sample()
        hxg = -dynamics.get_log_likelihood()
        return hxg


class DynamicsPredictionEntropyMetrics(Metrics):
    def eval(self, config: Config):
        dynamics_entropy = DynamicsPredictionEntropy(
            config=config,
            num_procs=config.get_value("num_procs", 1),
            seed=config.get_value("seed", int(time.time())) + self.counter,
        )
        self.counter += len(self.config)
        samples = dynamics_entropy.compute(
            config.metrics.dynamics_prediction_entropy.get_value("num_samples", 10)
        )

        return Statistics.compute(
            samples, error_type=config.metrics.dynamics_prediction_entropy.error_type
        )


if __name__ == "__main__":
    pass
