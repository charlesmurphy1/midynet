import matplotlib.pyplot as plt
import numpy as np

from random import sample
from _midynet.mcmc import DynamicsMCMC
from _midynet.mcmc.callbacks import CollectLikelihoodOnSweep
from _midynet.utility import get_edge_list, seedWithTime, seed
from _midynet.proposer import GraphMove
from midynet.config import (
    DynamicsFactory,
    RandomGraphFactory,
    RandomGraphMCMCFactory,
    ExperimentConfig,
    Wrapper,
)
from midynet.metrics.util import (
    get_log_evidence_exact,
    get_log_evidence_annealed,
    get_log_evidence_meanfield,
)


def makeConfig():
    c = ExperimentConfig.default("test", "ising", "ser")
    c.graph.set_value("size", 100)
    c.graph.edge_count.set_value("state", 250)
    c.graph.set_value("sample_graph_prior_prob", 0.0)
    c.graph.set_value("heterogeneity", 0.0)
    c.dynamics.set_value("num_steps", 100)
    c.dynamics.set_coupling(1.0)
    c.dynamics.set_value("recovery_prob", 0.5)
    c.insert("burn_per_vertex", 5)
    c.insert("num_betas", 10)
    c.insert("exp_betas", 0.5)
    c.insert("start_from_original", False)
    c.insert("initial_burn", 10000)
    c.insert("num_sweeps", 1000)
    return c


def makeMCMC(cfg):
    randomGraph = RandomGraphFactory.build(cfg.graph)
    dynamics = DynamicsFactory.build(cfg.dynamics)
    dynamics.set_random_graph(randomGraph.get_wrap())
    randomGraphMCMC = RandomGraphMCMCFactory.build(cfg.graph)
    mcmc = DynamicsMCMC(dynamics, randomGraphMCMC.get_wrap(), 1, 1, 0)
    return Wrapper(
        mcmc,
        randomGraph=randomGraph,
        dynamics=dynamics,
        randomGraphMCMC=randomGraphMCMC,
    )


def main():
    seed(1)
    cfg = makeConfig()
    mcmc = makeMCMC(cfg)
    callback = CollectLikelihoodOnSweep()
    mcmc.add_callback(callback)
    dynamics = mcmc.get_other("dynamics")
    dynamics.sample()
    original_graph = dynamics.get_graph()
    hg = dynamics.get_log_prior()
    hxg = dynamics.get_log_likelihood()
    if cfg.graph.size < 6:
        exact = hxg - get_log_evidence_exact(mcmc, cfg)
    else:
        exact = None
    annealed = hxg - get_log_evidence_annealed(mcmc, cfg, verbose=1)
    meanfield = hxg - get_log_evidence_meanfield(mcmc, cfg, verbose=1)
    print(f"{exact=}", f"{annealed=}", f"{meanfield=}", f"{hg=}")
    # if cfg.graph.size < 6:
    #     mf = get_log_posterior_exact_meanfield(mcmc, cfg)
    #     print(mf, exact, hg)
    #     plt.axhline(-mf, linestyle="--", label="MF")
    #     plt.axhline(-exact, linestyle="-.", label="Exact")
    # mcmc.set_up()
    # callback.collect()
    #
    # entropy = []
    # for i in range(1000):
    #     s, f = mcmc.do_MH_sweep(250)
    #     entropy.append(-callback.get_log_posterior_estimate(original_graph))
    #     if i % 10 == 0:
    #         print(i, s, f, entropy[-1])
    # print(entropy[-1], -hg)
    # plt.loglog(entropy, label="MCMC")
    # plt.axhline(-hg, linestyle="dotted", color="red", label=r"$H(G)$")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
