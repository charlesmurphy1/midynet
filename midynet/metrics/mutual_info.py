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
    get_log_posterior_meanfield,
    get_log_evidence_annealed,
    get_log_evidence_arithmetic,
    get_log_evidence_harmonic,
    get_log_evidence_exact,
    get_graph_log_evidence,
)

__all__ = ("MutualInformation", "MutualInformationMetrics")


@dataclass
class MutualInformation(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        utility.seed(seed)
        graph_model = RandomGraphFactory.build(self.config.graph_prior)
        data_model = DataModelFactory.build(self.config.data_model)
        data_model.set_graph_prior(graph_model)
        data_model.sample()

        metric_cf = self.config.metrics.mutual_info

        hxg = -data_model.get_log_likelihood()
        hg = -get_graph_log_evidence(graph_model, metric_cf)
        hbg = -data_model.get_log_prior() if graph_model.labeled else hg
        hgx = -get_log_posterior(data_model, metric_cf)
        hx = hg + hxg - hgx
        out = {
            "evidence": hx,
            "prior": hg,
            "likelihood": hxg,
            "posterior": hgx,
            "mutual-info": hg - hgx,
        }
        if graph_model.labeled:
            out["graph-joint"] = hbg
        if metric_cf.get_value("to_bits", True):
            out = {k: v / np.log(2) for k, v in out.items()}
        return out


class MutualInformationMetrics(Metrics):
    def eval(self, config: Config):
        mutual_info = MutualInformation(
            config=config,
            num_procs=config.get_value("num_procs", 1),
            seed=config.get_value("seed", int(time.time())),
        )

        samples = mutual_info.compute(
            config.metrics.mutual_info.get_value("num_samples", 10)
        )
        sample_dict = defaultdict(list)
        for s in samples:
            for k, v in s.items():
                sample_dict[k].append(v)
        res = {
            k: Statistics.compute(
                v, error_type=config.metrics.mutual_info.get("error_type", "std")
            )
            for k, v in sample_dict.items()
        }

        out = {f"{k}-{kk}": vv for k, v in res.items() for kk, vv in v.items()}
        print(out)
        return out


if __name__ == "__main__":
    pass
