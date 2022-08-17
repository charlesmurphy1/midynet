import time
import midynet

from dataclasses import dataclass, field
from midynet import utility
from midynet.config import (
    Config,
    RandomGraphFactory,
    DataModelFactory,
)
from .multiprocess import Expectation
from .metrics import Metrics
from .statistics import Statistics

__all__ = ("DynamicsPredictionEntropy", "DynamicsPredictionEntropyMetrics")


@dataclass
class DataPredictionEntropy(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        utility.seed(seed)
        graph_model = RandomGraphFactory.build(self.config.graph_prior)
        data_model = DataModelFactory.build(self.config.data_model)
        data_model.set_graph_prior(graph_model)
        data_model.sample()
        hxg = -data_model.get_log_likelihood()
        return hxg


class DataPredictionEntropyMetrics(Metrics):
    def eval(self, config: Config):
        data_entropy = DataPredictionEntropy(
            config=config,
            num_procs=config.get_value("num_procs", 1),
            seed=config.get_value("seed", int(time.time())) + self.counter,
        )
        self.counter += len(self.config)
        samples = data_entropy.compute(
            config.metrics.data_prediction_entropy.get_value("num_samples", 10)
        )

        return Statistics.compute(
            samples,
            error_type=config.metrics.data_prediction_entropy.error_type,
        )


if __name__ == "__main__":
    pass
