import time
import numpy as np

from typing import Dict, Tuple, Any
from graphinf.utility import seed as gi_seed
from basegraph import core as bg
from graphinf.graph import RandomGraphWrapper
from midynet.config import (
    GraphFactory,
    DataModelFactory,
)


from midynet.config import Config
from ..statistics import Statistics
from .metrics import ExpectationMetrics
from .reconinfo import (
    ReconstructionInformationMeasures,
)

__all__ = (
    "TargetedReconstructionInformationMeasures",
    "TargetedReconstructionInformationMeasuresMetrics",
)


class TargetedReconstructionInformationMeasures(ReconstructionInformationMeasures):
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
            if isinstance(target, bg.UndirectedMultigraph):
                g0 = target
            else:
                assert issubclass(target.__class__, RandomGraphWrapper)
                g0 = target.get_state()
        x0 = data_model.get_random_state(config.data_model.get("num_active", -1))
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

    def postprocess(self, stats: Dict[str, Statistics]) -> Dict[str, Statistics]:
        stats["recon"] = stats["mutualinfo"] / stats["prior"]
        stats["pred"] = stats["mutualinfo"] / stats["evidence"]
        return stats


if __name__ == "__main__":
    pass
