import time
import numpy as np

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, Tuple, Any
from basegraph.core import UndirectedMultigraph
from graphinf.random_graph import RandomGraphWrapper
from graphinf.utility import seed as gi_seed
from midynet.config import (
    GraphFactory,
    DataModelFactory,
)

from midynet.config import Config
from ..statistics import Statistics
from .metrics import ExpectationMetrics
from .multiprocess import Expectation
from .util import (
    get_log_posterior_meanfield,
    get_log_evidence,
    get_graph_log_evidence_meanfield,
    get_graph_log_evidence_annealed,
    get_graph_log_evidence_exact,
)
from .reconinfo import (
    ReconstructionInformationMeasures,
    ReconstructionInformationMeasuresMetrics,
)

__all__ = (
    "TargetedReconstructionInformationMeasures",
    "TargetedReconstructionInformationMeasuresMetrics",
)


class TargetedReconstructionInformationMeasures(
    ReconstructionInformationMeasures
):
    def setup(self, seed: int) -> Tuple[Config, Dict[str, Any]]:
        gi_seed(seed)
        config = Config.from_dict(self.params)
        prior = GraphFactory.build(config.prior)
        data_model = DataModelFactory.build(config.data_model)

        data_model.set_graph_prior(prior)
        if config.target == "None":
            prior.sample()
            g0 = prior.get_state()
        else:
            target = GraphFactory.build(config.target)
            if isinstance(target, UndirectedMultigraph):
                g0 = target
            else:
                assert issubclass(target.__class__, RandomGraphWrapper)
                g0 = target.get_state()
        x0 = data_model.get_random_state(
            config.data_model.get("num_active", -1)
        )
        data_model.set_graph(g0)
        data_model.sample_state(x0)
        return config, dict(data_model=data_model, prior=prior)


class TargetedReconstructionInformationMeasuresMetrics(ExpectationMetrics):
    shortname = "targeted_reconinfo"
    keys = [
        "prior",
        "likelihood",
        "posterior",
        "evidence",
        "recon",
        "pred",
    ]
    expectation_factory = TargetedReconstructionInformationMeasures

    def postprocess(
        self, stats: Dict[str, Statistics]
    ) -> Dict[str, Statistics]:
        stats["recon"] = stats["mutualinfo"] / stats["prior"]
        stats["pred"] = stats["mutualinfo"] / stats["evidence"]
        return stats

    # def eval(self, config: Config):
    #     metrics = TargetedReconstructionInformationMeasures(
    #         config=config,
    #         num_workers=config.get("num_workers", 1),
    #         seed=config.get("seed", int(time.time())),
    #     )

    #     samples = metrics.compute(config.metrics.get("num_samples", 10))
    #     sample_dict = defaultdict(list)
    #     for s in samples:
    #         for k, v in s.items():
    #             sample_dict[k].append(v)

    #     stats = {}
    #     for k, v in sample_dict.items():
    #         stats[k] = Statistics.from_samples(
    #             v, reduction=config.metrics.get("reduction", "normal"), name=k
    #         )

    #     stats["recon"] = stats["mutualinfo"] / stats["prior"]
    #     stats["pred"] = stats["mutualinfo"] / stats["evidence"]

    #     out = dict()
    #     for k, s in stats.items():
    #         for sk, sv in s.__data__.items():
    #             out[k + "_" + sk] = [sv]
    #     return out


if __name__ == "__main__":
    pass
