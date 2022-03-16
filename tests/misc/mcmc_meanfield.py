import matplotlib.pyplot as plt
import numpy as np

from random import sample
from _midynet.mcmc import DynamicsMCMC
from _midynet.mcmc.callbacks import CollectEdgeMultiplicityOnSweep
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
    get_log_posterior_exact,
    get_log_posterior_exact_meanfield,
    get_log_posterior_meanfield,
)


def makeConfig():
    c = ExperimentConfig.default("test", "sis", "nbinom_cm")
    c.graph.set_value("size", 100)
    c.graph.edge_count.set_value("state", 250)
    c.graph.set_value("sample_graph_prior_prob", 0.0)
    c.graph.set_value("heterogeneity", 0.0)
    c.dynamics.set_value("num_steps", 1000)
    c.dynamics.set_coupling(0.1)
    c.dynamics.set_value("recovery_prob", 0.5)
    c.dynamics.set_value("auto_infection_prob", 0.001)
    c.dynamics.set_value("initial_active", 1)
    c.insert("burn_per_vertex", 5)
    c.insert("num_betas", 10)
    c.insert("exp_betas", 0.5)
    c.insert("start_from_original", True)
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
    print(cfg.format())
    mcmc = makeMCMC(cfg)
    callback = CollectEdgeMultiplicityOnSweep()
    mcmc.add_callback(callback)
    dynamics = mcmc.get_other("dynamics")
    dynamics.sample()
    x = np.array(dynamics.get_past_states())
    plt.plot(x.mean(1))
    plt.show()
    original_graph = dynamics.get_graph()
    hg = dynamics.get_log_prior()
    hxg = dynamics.get_log_likelihood()
    if cfg.graph.size < 6:
        exact = get_log_posterior_exact(mcmc, cfg)
        exact_meanfield = get_log_posterior_exact_meanfield(mcmc, cfg)
        plt.axhline(
            -exact, linestyle="-", color="blue", label=r"$H(G|X)$"
        )
        plt.axhline(
            -exact_meanfield,
            linestyle="--",
            color="blue",
            label=r"$H_{MF}(G|X)$",
        )
    else:
        exact = None
        exact_meanfield = None
    callback.collect()
    dynamics.sample_graph()
    mcmc.set_up()
    gmcmc = mcmc.get_random_graph_mcmc()
    edge_proposer = gmcmc.get_edge_proposer()
    entropy = []
    logPosterior = []
    # s, f = mcmc.do_MH_sweep(10000)
    s, f = 0, 0
    for i in range(200):
        # s, f = mcmc.do_MH_sweep(250)
        for j in range(250):
            move = edge_proposer.propose_move()
            logp = dynamics.get_log_joint_ratio_from_graph_move(move)
            if np.random.rand() < np.exp(logp):
                s += 1
                dynamics.apply_graph_move(move)
                gmcmc.apply_graph_move(move)
            else:
                f += 1
        callback.collect()
        logPosterior.append(callback.get_log_posterior_estimate(original_graph))
        entropy.append(callback.get_marginal_entropy())
        if i % 10 == 0:
            print(i, s, f, entropy[-1])
    meanfield = entropy[-1]
    plt.semilogx(entropy, label="MCMC")
    plt.axhline(-hg, linestyle="dotted", color="red", label=r"$H(G)$")
    plt.legend()
    plt.show()
    print(f"{exact=}", f"{exact_meanfield=}", f"{entropy[-1]=}", f"{logPosterior[-1]=}", f"{hg=}")


if __name__ == "__main__":
    main()
