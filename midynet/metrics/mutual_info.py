import time
from collections import defaultdict
from dataclasses import dataclass, field

from midynet.config import *
from _midynet import utility
from _midynet.mcmc import DynamicsMCMC
from .multiprocess import MultiProcess, Expectation
from .metrics import ExpectationMetrics
from .util import get_log_evidence

__all__ = ["MutualInformation", "MutualInformationMetrics"]


@dataclass
class MutualInformation(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        utility.seed(seed)
        graph = RandomGraphFactory.build(self.config.graph)
        dynamics = DynamicsFactory.build(self.config.dynamics)
        dynamics.set_random_graph(graph.get_wrap())
        random_graph_mcmc = RandomGraphMCMCFactory.build(self.config.graph)
        mcmc = DynamicsMCMC(dynamics, random_graph_mcmc.get_wrap())
        mcmc.sample()
        hx = -get_log_evidence(mcmc, self.config.metrics.mutualinfo)
        hg = -mcmc.get_log_prior()
        hxg = -mcmc.get_log_likelihood()
        hgx = hg + hxg - hx
        return {"hx": hx, "hg": hg, "hxg": hxg, "hgx": hgx}


class MutualInformationMetrics(ExpectationMetrics):
    def __post_init__(self):
        self.statistics = MCStatistics(
            self.config.metrics.mutualinfo.get_value("error_type", "std")
        )

    def eval(self, config: Config):
        mutual_info = MutualInformation(
            config=config,
            num_procs=config.metrics.get_value("num_procs", 1),
            seed=config.metrics.get_value("seed", int(time.time())),
        )
        samples = mutual_info.compute(config.metrics.get_value("num_samples", 10))
        sample_dict = defaultdict(list)
        for s in samples:
            for k, v in s.items():
                sample_dict[k].append(v)
        result_dict = {k: self.statistics(v) for k, v in sample_dict.items()}
        return {f"{k}-{kk}": vv for k, v in result_dict.items() for kk, vv in v.items()}


if __name__ == "__main__":
    pass
