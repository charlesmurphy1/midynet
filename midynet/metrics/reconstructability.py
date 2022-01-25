import time
from dataclasses import dataclass, field

from midynet.config import *
from _midynet import utility
from _midynet.mcmc import DynamicsMCMC
from .multiprocess import MultiProcess, Expectation
from .metrics import ExpectationMetrics
from .util import get_log_posterior

__all__ = ["Reconstructability", "ReconstructabilityMetrics"]


@dataclass
class Reconstructability(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        utility.seed(seed)
        graph = RandomGraphFactory.build(self.config.graph)
        dynamics = DynamicsFactory.build(self.config.dynamics)
        dynamics.set_random_graph(graph.get_wrap())
        random_graph_mcmc = RandomGraphMCMCFactory.build(self.config.graph)

        mcmc = DynamicsMCMC(dynamics, random_graph_mcmc.get_wrap())
        dynamics.sample()
        hg = -graph.get_log_likelihood()
        hgx = -get_log_posterior(mcmc, self.config.metrics.reconstructability)

        return (hg - hgx) / hg


class ReconstructabilityMetrics(ExpectationMetrics):
    def __post_init__(self):
        self.statistics = MCStatistics(
            self.config.metrics.reconstructability.get_value("error_type", "std")
        )

    def eval(self, config: Config):
        reconstructability = Reconstructability(
            config=config,
            num_procs=config.get_value("num_procs", 1),
            seed=config.get_value("seed", int(time.time())),
        )
        samples = reconstructability.compute(
            config.metrics.get_value("num_samples", 10)
        )
        return self.statistics(samples)


if __name__ == "__main__":
    pass
