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
    "ReconstructionEfficiency",
    "ReconstructionEfficiencyMetrics",
)


class ReconstructionEfficiency(ReconstructionInformationMeasures):
    def setup(self, seed: int) -> Tuple[Config, Dict[str, Any]]:
        gi_seed(seed)
        config = Config.from_dict(self.params)
        prior = GraphFactory.build(config.prior)
        model = DataModelFactory.build(config.data_model)

        model.set_graph_prior(prior)
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
        if "num_active" in config.data_model:
            x0 = model.get_random_state(config.data_model.get("num_active", -1))
        prior.from_graph(g0)
        
        if config.metrics.resample_graph:
            prior.sample_state() # Only samples the graph, with its parameters fixed.

        if "num_active" in config.data_model:
            x0 = model.get_random_state(config.data_model.get("num_active", -1))
            model.sample_state(x0)
        else:
            model.sample_state()
        return config, dict(model=model, prior=prior)



class ReconstructionEfficiencyMetrics(ExpectationMetrics):
    shortname = "recon_efficiency"
    keys = [
        "prior",
        "likelihood",
        "posterior",
        "evidence",
        "recon",
        "pred",
    ]
    expectation_factory = ReconstructionEfficiency

    def postprocess(self, samples: Dict[str, Statistics]) -> Dict[str, Statistics]:
        stats = self.reduce(samples, self.configs.metrics.get("reduction", "normal"))
        stats["recon"] = stats["mutualinfo"] / stats["prior"]
        stats["pred"] = stats["mutualinfo"] / stats["evidence"]
        out = self.format(stats)
        print(out)
        return out



if __name__ == "__main__":
    pass
