import time
from dataclasses import dataclass, field
from _midynet import utility
from midynet.config import Config, MCMCFactory
from .metrics import Metrics
from .multiprocess import Expectation
from .statistics import Statistics
from .util import get_log_posterior

__all__ = ("Reconstructability", "ReconstructabilityMetrics")


@dataclass
class Reconstructability(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        utility.seed(seed)
        mcmc = MCMCFactory.build_reconstruction(self.config)
        mcmc.sample()
        mcmc.set_up()
        hg = -mcmc.get_log_prior()
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
