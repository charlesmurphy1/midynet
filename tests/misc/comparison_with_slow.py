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

from slowmidynet.graphs import NBinomGraph
from slowmidynet.dynamics import SISDynamics


def makeConfig():
    c = ExperimentConfig.default("test", "sis", "nbinom_cm")
    c.graph.set_value("size", 5)
    c.graph.edge_count.set_value("state", 5)
    c.graph.set_value("sample_graph_prior_prob", 0.0)
    c.graph.set_value("heterogeneity", 0.001)
    c.dynamics.set_value("num_steps", 10)
    c.dynamics.set_coupling(0.1)
    c.dynamics.set_value("recovery_prob", 0.5)
    c.dynamics.set_value("auto_infection_prob", 0.001)
    c.dynamics.set_value("num_active", 5)
    c.insert("burn_per_vertex", 5)
    c.insert("num_betas", 10)
    c.insert("exp_betas", 0.5)
    c.insert("start_from_original", True)
    c.insert("initial_burn", 10000)
    c.insert("num_sweeps", 1000)
    return c


def makeSlowModel(cfg):
    graph = NBinomGraph(
        cfg.graph.size, cfg.graph.edge_count.state, cfg.graph.heterogeneity
    )
    dynamics = SISDynamics(
        graph,
        alpha=cfg.dynamics.infection_prob,
        beta=cfg.dynamics.recovery_prob,
        epsilon=cfg.dynamics.auto_infection_prob,
        normalize=cfg.dynamics.normalize,
        init_num_infected=cfg.dynamics.num_active,
    )
    return dynamics


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


def get_slow_log_posterior_estimate(model, original_graph):
    logP = 0
    c = model.cg.gp.count
    for e in model.cg.edges():
        w = np.array(model.cg.ep.w[e]).astype("float")
        wc = np.array(model.cg.ep.wcount[e]).astype("float")
        p = wc / c
        p0 = 1 - p.sum()
        ec = original_graph.ep.ec[e]
        # print(f"{p0=}, {p=}, {w=}, {wc=}")
        if p0 > 0 and ec == 0:
            logP += np.log(p0)
        else:
            index = np.where(w == ec)[0]
            # print(w, wc, e, ec, p)
            logP += np.log(p[index])
    return logP


def main():

    seed(1)
    cfg = makeConfig()
    mcmc = makeMCMC(cfg)
    slowDynamics = makeSlowModel(cfg)
    slowDynamics.sample(num_steps=cfg.dynamics.num_steps)
    slow_og_graph = slowDynamics.get_params()["graph"].copy()

    callback = CollectEdgeMultiplicityOnSweep()
    mcmc.add_callback(callback)
    dynamics = mcmc.get_other("dynamics")
    dynamics.sample()
    original_graph = dynamics.get_graph()
    hg = dynamics.get_log_prior()
    hxg = dynamics.get_log_likelihood()
    if cfg.graph.size < 6:
        exact = get_log_posterior_exact(mcmc, cfg)
        exact_meanfield = get_log_posterior_exact_meanfield(mcmc, cfg)
        plt.axhline(-exact, linestyle="-", color="blue", label=r"$H(G|X)$")
        plt.axhline(
            -exact_meanfield,
            linestyle="--",
            color="blue",
            label=r"$H_{MF}(G|X)$",
        )
    else:
        exact = None
        exact_meanfield = None

    slowDynamics.collect()

    # print("Original: ")
    # for e in slow_og_graph.edges():
    #     ec = slow_og_graph.ep.ec[e]
    #     e = (int(e.source()), int(e.target()))
    #     print(f"{e=}, {ec=}")
    #
    # print("\n Collected: ")
    # c = slowDynamics.graph.cg.gp.count
    # for e in slowDynamics.graph.cg.edges():
    #
    #     w = np.array(slowDynamics.graph.cg.ep.w[e]).astype("float")
    #     wc = np.array(slowDynamics.graph.cg.ep.wcount[e]).astype("float")
    #     e = (int(e.source()), int(e.target()))
    #     print(f"{e=}, {w=}, {wc=}, {c=}")

    callback.collect()

    dynamics.sample_graph()
    mcmc.set_up()
    gmcmc = mcmc.get_random_graph_mcmc()
    edge_proposer = gmcmc.get_edge_proposer()
    entropy = []
    logPosterior = []
    slow_logPosterior = []
    # s, f = mcmc.do_MH_sweep(10000)
    s, f = 0, 0
    for i in range(1000):
        _s, _f = mcmc.do_MH_sweep(250)
        s += _s
        f += _f
        for j in range(250):
            move = slowDynamics.propose_move()
            dS = slowDynamics.virtual_move(move)
            if np.random.rand() < np.exp(dS):
                slowDynamics.make_move(move)
        slowDynamics.collect()
        slow_logPosterior.append(
            get_slow_log_posterior_estimate(slowDynamics.graph, slow_og_graph)
        )
        logPosterior.append(callback.get_log_posterior_estimate(original_graph))
        entropy.append(callback.get_marginal_entropy())
        if i % 10 == 0:
            print(i, s, f, entropy[-1], logPosterior[-1], slow_logPosterior[-1])
    plt.semilogx([-p for p in slow_logPosterior], label="slow-MCMC")
    plt.semilogx([-p for p in logPosterior], label="MCMC")
    plt.axhline(-hg, linestyle="dotted", color="red", label=r"$H(G)$")
    plt.legend()
    plt.show()
    print(
        f"{exact=}",
        f"{exact_meanfield=}",
        f"{entropy[-1]=}",
        f"{logPosterior[-1]=}",
        f"{hg=}",
    )


if __name__ == "__main__":
    main()
