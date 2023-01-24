import time
import numpy as np

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Tuple, Dict, Any
from basegraph.core import UndirectedMultigraph
from graphinf.random_graph import RandomGraphWrapper
from graphinf.utility import seed as gi_seed
from midynet.config import (
    GraphFactory,
    DataModelFactory,
)

from midynet.config import Config
from midynet.statistics import Statistics
from .metrics import ExpectationMetrics
from .multiprocess import Expectation
from .util import (
    get_log_posterior_meanfield,
    get_log_evidence,
    get_graph_log_evidence_meanfield,
    get_graph_log_evidence_annealed,
    get_graph_log_evidence_exact,
)

__all__ = ("ReconstructionInformation", "ReconstructionInformationMetrics")


class ReconstructionInformationMeasures(Expectation):
    def __init__(self, config: Config, **kwargs):
        self.params = config.dict
        super().__init__(**kwargs)

    def setup(self, seed: int) -> Tuple[Config, Dict[str, Any]]:
        gi_seed(seed)
        config = Config.from_dict(self.params)
        prior = GraphFactory.build(config.prior)
        data_model = DataModelFactory.build(config.data_model)

        data_model.set_graph_prior(prior)
        prior.sample()
        g0 = prior.get_state()
        x0 = data_model.get_random_state(
            config.data_model.get("num_active", -1)
        )
        data_model.set_graph(g0)
        data_model.sample_state(x0)

        return config, dict(data_model=data_model, prior=prior)

    def gather(self, data_model, config):
        method = config.metrics.get("method", "meanfield")
        graph_evidence_method = config.metrics.get(
            "graph_evidence_method", method
        )

        og = data_model.get_graph()

        if not data_model.graph_prior.labeled:
            prior = -data_model.graph_prior.get_log_joint()
        elif graph_evidence_method == "exact":
            prior = -get_graph_log_evidence_exact(
                data_model.graph_prior, config.metrics
            )
        elif graph_evidence_method == "annealed":
            prior = -get_graph_log_evidence_annealed(
                data_model.graph_prior, config.metrics
            )
        else:
            prior = -get_graph_log_evidence_meanfield(
                data_model.graph_prior, config.metrics
            )
        data_model.set_graph(og)
        likelihood = -data_model.get_log_likelihood()

        if method == "meanfield":
            posterior = -get_log_posterior_meanfield(
                data_model, config.metrics
            )
            evidence = prior + likelihood - posterior
        else:
            evidence = -get_log_evidence(data_model, config.metrics)
            posterior = prior + likelihood - evidence

        data_model.set_graph(og)
        return dict(
            prior=prior,
            likelihood=likelihood,
            posterior=posterior,
            evidence=evidence,
        )

    def func(self, seed: int) -> float:
        config, model_dict = self.setup(seed)
        data_model, prior = model_dict["data_model"], model_dict["prior"]
        out = self.gather(data_model, config)
        out["mutualinfo"] = out["prior"] - out["posterior"]

        if prior.labeled:
            out["graph_joint"] = prior.get_log_joint()
            out["graph_prior"] = prior.get_label_log_joint()
            out["graph_evidence"] = -out["prior"]
            out["graph_posterior"] = (
                out["graph_joint"] - out["graph_evidence"]
            )
        if config.metrics.get("to_bits", True):
            out = {k: v / np.log(2) for k, v in out.items()}
        print(out)
        return out


class ReconstructionInformationMeasuresMetrics(ExpectationMetrics):
    shortname = "reconinfo"
    keys = [
        "prior",
        "likelihood",
        "posterior",
        "evidence",
        "mutualinfo",
        "recon",
        "pred",
    ]
    expectation_factory = ReconstructionInformationMeasures

    def postprocess(
        self, stats: Dict[str, Statistics]
    ) -> Dict[str, Statistics]:
        stats["recon"] = stats["mutualinfo"] / stats["prior"]
        stats["pred"] = stats["mutualinfo"] / stats["evidence"]
        return stats


if __name__ == "__main__":
    pass
