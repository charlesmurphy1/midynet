import numpy as np
import time
from dataclasses import dataclass, field

from midynet.config import *
from _midynet import utility
from .multiprocess import MultiProcess, Expectation
from .metrics import ExpectationMetrics

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


class DynamicsPredictionEntropyMetrics(ExpectationMetrics):
    def __post_init__(self):
        self.statistics = MCStatistics(
            self.config.metrics.dynamics_predcition_entropy.get_value(
                "error_type", "std"
            )
        )

    def eval(self, config: Config):
        dynamics_entropy = DynamicsPredictionEntropy(
            config=config,
            num_procs=config.metrics.get_value("num_procs", 1),
            seed=config.metrics.get_value("seed", int(time.time())) + self.counter,
        )
        self.counter += len(self.config)
        samples = dynamics_entropy.compute(
            config.metrics.dynamics_prediction_entropy.get_value("num_samples", 10)
        )
        return self.statistics(samples)


if __name__ == "__main__":
    pass
