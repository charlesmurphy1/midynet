import time

from dataclasses import dataclass, field
from _midynet import utility
from midynet.config import (
    Config,
    RandomGraphFactory,
    DataModelFactory,
    ReconstructionMCMC,
)
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
        graph = RandomGraphFactory.build(config.graph)
        data_model = DataModelFactory.build(config.data_model)
        data_model.set_graph_prior(graph)
        mcmc = ReconstructionMCMC(data_model, graph)
        mcmc.sample()
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
