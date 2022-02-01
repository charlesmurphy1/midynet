import time
from dataclasses import dataclass, field

from _midynet.mcmc import DynamicsMCMC

from _midynet import utility
from midynet.config import *

from .metrics import Metrics
from .multiprocess import Expectation, MultiProcess
from .statistics import Statistics
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
        mcmc = DynamicsMCMC(
            dynamics,
            random_graph_mcmc.get_wrap(),
            1,
            1,
            self.config.graph.sample_graph_prior_prob,
        )
        dynamics.sample()
        hg = -graph.get_log_likelihood()
        hgx = -get_log_posterior(mcmc, self.config.metrics.reconstructability)

        return (hg - hgx) / hg


class ReconstructabilityMetrics(Metrics):
    def eval(self, config: Config):
        reconstructability = Reconstructability(
            config=config,
            num_procs=config.get_value("num_procs", 1),
            seed=config.get_value("seed", int(time.time())),
        )
        samples = reconstructability.compute(
            config.metrics.reconstructability.get_value("num_samples", 10)
        )
        return Statistics.compute(
            samples, error_type=config.metrics.reconstructability.error_type
        )


if __name__ == "__main__":
    pass
