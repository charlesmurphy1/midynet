import time
from dataclasses import dataclass, field

from midynet.config import *
from midynet import utility
from .multiprocess import MultiProcess, Expectation
from .metrics import ExpectationMetrics
from .mcmc_functions import build_dynamics_mcmc

__all__ = ["DynamicsEntropy", "DynamicsEntropyMetrics"]


@dataclass
class DynamicsEntropy(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        utility.seed(seed)
        graph = RandomGraphFactory.build(self.config.graph)
        dynamics = DynamicsFactory.build(self.config.dynamics)
        dynamics.set_random_graph(graph.get_wrap())
        random_graph_mcmc = RandomGraphMCMCFactory.build(self.config.graph)
        mcmc = DynamicsMCMC(dynamics, random_graph_mcmc)
        mcmc.sample()
        return -get_log_evidence(mcmc, self.config.metrics)


class DynamicsEntropyMetrics(ExpectationMetrics):
    def eval(self, config: Config):
        dynamics_entropy = DynamicsEntropy(
            config=config,
            num_procs=config.metrics.get_value("num_procs", 1),
            seed=config.metrics.get_value("seed", int(time.time())),
        )
        return dynamics_entropy.compute(config.metrics.get_value("num_samples", 10))


if __name__ == "__main__":
    pass
