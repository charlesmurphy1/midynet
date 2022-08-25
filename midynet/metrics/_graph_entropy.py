import time

from dataclasses import dataclass, field
from midynet import utility
from midynet.random_graph import BlockLabeledRandomGraph
from midynet.config import (
    Config,
    RandomGraphFactory,
)
from .metrics import Metrics
from .multiprocess import Expectation
from .statistics import Statistics
from .util import get_graph_log_evidence

__all__ = ("GraphEntropy", "GraphEntropyMetrics")


@dataclass
class GraphEntropy(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        utility.seed(seed)
        graph_model = RandomGraphFactory.build(self.config.graph_prior)
        graph_model.sample()
        S = -get_graph_log_evidence(graph_model, self.config.metrics.graph_entropy)
        return S


class GraphEntropyMetrics(Metrics):
    def eval(self, config: Config):
        graph_entropy = GraphEntropy(
            config=config,
            num_procs=config.get_value("num_procs", 1),
            seed=config.get_value("seed", int(time.time())),
        )
        samples = graph_entropy.compute(
            config.metrics.graph_entropy.get_value("num_samples", 10)
        )

        return Statistics.compute(
            samples,
            error_type=config.metrics.graph_entropy.get_value("error_type", "std"),
        )


if __name__ == "__main__":
    pass
