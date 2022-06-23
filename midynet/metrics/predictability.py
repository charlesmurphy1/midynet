import time
from dataclasses import dataclass, field
from _midynet import utility
from midynet.config import Config, MCMCFactory
from .metrics import Metrics
from .multiprocess import Expectation
from .statistics import Statistics
from .util import get_log_evidence

__all__ = ("Predictability", "PredictabilityMetrics")


@dataclass
class Predictability(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        utility.seed(seed)
        mcmc = MCMCFactory.build_reconstruction(self.config)
        mcmc.sample()
        mcmc.set_up()
        hxg = -mcmc.get_log_likelihood()
        hx = -get_log_evidence(mcmc, self.config.metrics.predictability)
        return (hx - hxg) / hx


class PredictabilityMetrics(Metrics):
    def eval(self, config: Config):
        predictability = Predictability(
            config=config,
            num_procs=config.get_value("num_procs", 1),
            seed=config.get_value("seed", int(time.time())),
        )
        samples = predictability.compute(
            config.metrics.predictability.get_value("num_samples", 10)
        )
        return Statistics.compute(
            samples, error_type=config.metrics.predictability.error_type
        )


if __name__ == "__main__":
    pass
