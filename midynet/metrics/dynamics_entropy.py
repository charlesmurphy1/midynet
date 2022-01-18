import time
import numpy as np
from dataclasses import dataclass, field

from midynet.config import *
from midynet import utility
from _midynet.mcmc import DynamicsMCMC
from .multiprocess import MultiProcess, Expectation
from .metrics import ExpectationMetrics
from .util import get_log_evidence

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
        mcmc = DynamicsMCMC()
        mcmc.set_dynamics(dynamics)
        mcmc.set_random_graph_mcmc(random_graph_mcmc.get_wrap())
        # print(
        #     "HERE",
        #     id(mcmc.get_dynamics().get_random_graph())
        #     == id(random_graph_mcmc.get_random_graph()),
        # )
        mcmc.sample()
        print(mcmc.get_dynamics().get_graph())
        # return -get_log_evidence(mcmc, self.config.metrics.dynamics_entropy)
        return 0


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
