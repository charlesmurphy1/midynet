import time
import numpy as np

from dataclasses import dataclass, field
from collections import defaultdict
from midynet import utility
from midynet.random_graph import BlockLabeledRandomGraph
from midynet.config import (
    Config,
    RandomGraphFactory,
    DataModelFactory,
)
from .metrics import Metrics
from .multiprocess import Expectation
from .statistics import Statistics
from .util import (
    get_log_evidence,
    get_log_posterior,
    get_log_prior_meanfield,
    get_posterior_entropy_partition_meanfield,
)

__all__ = ("MutualInformation", "MutualInformationMetrics")


@dataclass
class MutualInformation(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        utility.seed(seed)
        graph_model = RandomGraphFactory.build(self.config.graph)
        data_model = DataModelFactory.build(self.config.data_model)
        data_model.set_graph_prior(graph_model)
        data_model.sample()

        hxg = -data_model.get_log_likelihood() / np.log(2)
        hg = -data_model.get_log_prior() / np.log(2)
        if issubclass(graph_model.__class__, BlockLabeledRandomGraph):
            hg += get_posterior_entropy_partition_meanfield(
                graph_model, self.config.metrics.mutualinfo
            ) / np.log(2)
        hx = -get_log_evidence(
            data_model, self.config.metrics.mutualinfo, verbose=0
        ) / np.log(2)
        hgx = hg + hxg - hx
        mi = hg - hgx
        out = {"hx": hx, "hg": hg, "hxg": hxg, "hgx": hgx, "mi": mi}
        return out


class MutualInformationMetrics(Metrics):
    def eval(self, config: Config):
        mutual_info = MutualInformation(
            config=config,
            num_procs=config.get_value("num_procs", 1),
            seed=config.get_value("seed", int(time.time())),
        )

        samples = mutual_info.compute(
            config.metrics.mutualinfo.get_value("num_samples", 10)
        )
        sample_dict = defaultdict(list)
        for s in samples:
            for k, v in s.items():
                sample_dict[k].append(v)
        res = {
            k: Statistics.compute(v, error_type=config.metrics.mutualinfo.error_type)
            for k, v in sample_dict.items()
        }

        out = {f"{k}-{kk}": vv for k, v in res.items() for kk, vv in v.items()}
        return out


if __name__ == "__main__":
    pass
