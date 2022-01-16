import time
from dataclasses import dataclass, field

from midynet.config import *
from midynet import utility
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
        return -dynamics.get_log_likelihood()


class DynamicsPredictionEntropyMetrics(ExpectationMetrics):
    def eval(self, config: Config):
        dynamics_entropy = DynamicsPredictionEntropy(
            config=config,
            num_procs=config.metrics.get_value("num_procs", 1),
            seed=config.metrics.get_value("seed", int(time.time())),
        )
        samples = dynamics_entropy.compute(config.metrics.get_value("num_samples", 10))
        return self.statistics(samples)


if __name__ == "__main__":
    pass
