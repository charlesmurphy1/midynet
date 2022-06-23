import time

from dataclasses import dataclass, field
from _midynet import utility
from midynet.config import Config, MCMCFactory
from .metrics import Metrics
from .multiprocess import Expectation
from .statistics import Statistics
from .util import get_log_posterior

__all__ = ("GraphReconstructionEntropy", "GraphReconstructionEntropyMetrics")


@dataclass
class GraphReconstructionEntropy(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        utility.seed(seed)
        mcmc = MCMCFactory.build_reconstruction(self.config)
        mcmc.sample()
        mcmc.set_up()
        hgx = -get_log_posterior(mcmc, self.config.metrics.graph_reconstruction_entropy)
        return hgx


class GraphReconstructionEntropyMetrics(Metrics):
    def eval(self, config: Config):
        reconstruction_entropy = GraphReconstructionEntropy(
            config=config,
            num_procs=config.get_value("num_procs", 1),
            seed=config.get_value("seed", int(time.time())),
        )
        samples = reconstruction_entropy.compute(
            config.metrics.graph_reconstruction_entropy.get_value("num_samples", 10)
        )

        return Statistics.compute(
            samples,
            error_type=config.metrics.graph_reconstruction_entropy.error_type,
        )


if __name__ == "__main__":
    pass
