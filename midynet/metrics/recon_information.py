import time
import numpy as np

from dataclasses import dataclass, field
from collections import defaultdict
from graphinf.utility import seed as gi_seed
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
    get_log_evidence,
    get_graph_log_evidence_meanfield,
    get_graph_log_evidence_annealed,
    get_graph_log_evidence_exact,
)
from graphinf.random_graph import ErdosRenyiModel

__all__ = ("MutualInformation", "MutualInformationMetrics")


@dataclass
class ReconstructionInformationMeasures(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        gi_seed(seed)
        graph_model = RandomGraphFactory.build(self.config.graph_prior)
        data_model = DataModelFactory.build(self.config.data_model)
        metrics_cf = self.config.metrics.recon_information
        method = metrics_cf.get_value("method", "meanfield")
        graph_evidence_method = metrics_cf.get_value("graph_evidence_method", method)

        data_model.set_graph_prior(graph_model)
        data_model.sample()

        if not graph_model.labeled:
            hg = -graph_model.get_log_joint()
        elif graph_evidence_method == "exact":
            hg = -get_graph_log_evidence_exact(graph_model, metrics_cf)
        elif graph_evidence_method == "annealed":
            hg = -get_graph_log_evidence_annealed(graph_model, metrics_cf)
        else:
            hg = -get_graph_log_evidence_meanfield(graph_model, metrics_cf)

        hxg = -data_model.get_log_likelihood()
        if method == "meanfield":
            hgx = -get_log_posterior_meanfield(data_model, metrics_cf)
            hx = hg + hxg - hgx
        else:
            hx = -get_log_evidence(data_model, metrics_cf)
            hgx = hg + hxg - hx
        out = {
            "evidence": hx,
            "prior": hg,
            "likelihood": hxg,
            "posterior": hgx,
            "mutualinfo": hg - hgx,
        }

        if graph_model.labeled:
            out["graph_joint"] = graph_model.get_log_joint()
            out["graph_prior"] = graph_model.get_label_log_joint()
            out["graph_evidence"] = -hg
            out["graph_posterior"] = out["graph_joint"] - out["graph_evidence"]
        if metrics_cf.get_value("to_bits", True):
            out = {k: v / np.log(2) for k, v in out.items()}
        # print(out, graph_model.params)
        return out


class ReconstructionInformationMeasuresMetrics(Metrics):
    def eval(self, config: Config):
        metrics = ReconstructionInformationMeasures(
            config=config,
            num_procs=config.get_value("num_procs", 1),
            seed=config.get_value("seed", int(time.time())),
        )

        samples = metrics.compute(
            config.metrics.recon_information.get_value("num_samples", 10)
        )
        sample_dict = defaultdict(list)
        for s in samples:
            for k, v in s.items():
                sample_dict[k].append(v)
        res = {
            k: Statistics.compute(
                v,
                error_type=config.metrics.recon_information.get_value(
                    "error_type", "std"
                ),
            )
            for k, v in sample_dict.items()
        }

        out = {f"{k}-{kk}": vv for k, v in res.items() for kk, vv in v.items()}
        print(out)
        return out


if __name__ == "__main__":
    pass
