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
        mcmc = DynamicsMCMC()
        mcmc.set_dynamics(dynamics)
        mcmc.set_random_graph_mcmc(random_graph_mcmc.get_wrap())
        mcmc.sample()
        return dynamics.get_log_likelihood() - get_log_evidence(
            mcmc, self.config.metrics
        )


class ReconstructabilityMetrics(ExpectationMetrics):
    def eval(self, config: Config):
        reconstructability = Reconstructability(
            config=config,
            num_procs=config.metrics.get_value("num_procs", 1),
            seed=config.metrics.get_value("seed", int(time.time())),
        )
        samples = reconstructability.compute(
            config.metrics.get_value("num_samples", 10)
        )
        return self.statistics(samples)


if __name__ == "__main__":
    pass
