import time
from dataclasses import dataclass, field

from _midynet import utility
from midynet.config import *

from .metrics import Metrics
from .multiprocess import Expectation, MultiProcess
from .statistics import Statistics

__all__ = ["GraphEntropy", "GraphEntropyMetrics"]


@dataclass
class GraphEntropy(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        utility.seed(seed)
        graph = RandomGraphFactory.build(self.config.graph)
        graph.sample()
        hg = -graph.get_log_likelihood()
        return hg


class GraphEntropyMetrics(Metrics):
    def eval(self, config: Config):
        dynamics_entropy = GraphEntropy(
            config=config,
            num_procs=config.get_value("num_procs", 1),
            seed=config.get_value("seed", int(time.time())),
        )
        samples = dynamics_entropy.compute(
            config.metrics.graph_entropy.get_value("num_samples", 10)
        )

        return Statistics.compute(
            samples, error_type=config.metrics.graph_entropy.error_type
        )


if __name__ == "__main__":
    pass
