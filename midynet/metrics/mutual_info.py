import time
import numpy as np

from dataclasses import dataclass, field
from collections import defaultdict
from _midynet.mcmc import DynamicsMCMC
from _midynet import utility
from midynet.config import (
    Config,
    DynamicsFactory,
    RandomGraphFactory,
    RandomGraphMCMCFactory,
)
from .metrics import Metrics
from .multiprocess import Expectation
from .statistics import Statistics
from .util import get_log_evidence, get_log_posterior, get_log_prior_meanfield

__all__ = ("MutualInformation", "MutualInformationMetrics")


@dataclass
class MutualInformation(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        utility.seed(seed)
        graph = RandomGraphFactory.build(self.config.graph)
        dynamics = DynamicsFactory.build(self.config.dynamics)
        dynamics.set_random_graph(graph.get_wrap())
        random_graph_mcmc = RandomGraphMCMCFactory.build(self.config.graph)
        mcmc = DynamicsMCMC(
            dynamics,
            random_graph_mcmc.get_wrap(),
            1,
            1,
            self.config.graph.sample_graph_prior_prob,
        )
        dynamics.sample()
        hxg = -dynamics.get_log_likelihood() / np.log(2)
        if self.config.metrics.mutualinfo.method == "full-meanfield":
            hg = -get_log_prior_meanfield(
                mcmc, self.config.metrics.mutualinfo
            ) / np.log(2)

            hgx = -get_log_posterior(
                mcmc, self.config.metrics.mutualinfo, verbose=0
            ) / np.log(2)
            hx = hg + hxg - hgx
            print(hx, hxg, hg, hgx)
        else:
            hg = -dynamics.get_log_prior() / np.log(2)
            hx = -get_log_evidence(
                mcmc, self.config.metrics.mutualinfo, verbose=0
            ) / np.log(2)
            hxg = -dynamics.get_log_likelihood() / np.log(2)
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
            k: Statistics.compute(
                v, error_type=config.metrics.mutualinfo.error_type
            )
            for k, v in sample_dict.items()
        }

        out = {f"{k}-{kk}": vv for k, v in res.items() for kk, vv in v.items()}
        print(out)
        return out


if __name__ == "__main__":
    pass
