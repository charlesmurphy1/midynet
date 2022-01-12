import time
from dataclasses import dataclass, field

from midynet.config import *
from midynet import utility
from .multiprocess import MultiProcess, Expectation
from .metrics import Metrics

__all__ = ["Reconstructability", "ReconstructabilityMetrics"]


@dataclass
class Reconstructability(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        utility.seed(seed)
        graph = RandomGraphFactory.build(self.config.graph)
        dynamics = DynamicsFactory.build(self.config.dynamics)
        dynamics.set_random_graph(graph.get_wrap())
        raise NotImplementedError()


class ReconstructabilityMetrics(Metrics):
    def eval(self, config: Config):
        dynamics_entropy = Reconstructability(
            config=config,
            num_procs=config.metrics.get_value("num_procs", 1),
            seed=config.metrics.get_value("seed", int(time.time())),
        )
        return dynamics_entropy.compute(config.metrics.get_value("num_samples", 10))


if __name__ == "__main__":
    pass
