import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import loggamma

from netrd.distance import Frobenius
from _midynet.mcmc import DynamicsMCMC
from _midynet.mcmc.callbacks import CollectEdgeMultiplicityOnSweep
from _midynet.utility import get_edge_list
from midynet.config import (
    DynamicsFactory,
    RandomGraphFactory,
    RandomGraphMCMCFactory,
    ExperimentConfig,
    Wrapper,
)
from midynet.util import MCMCConvergenceAnalysis, enumerate_all_graphs
from midynet.metrics.util import (
    get_log_posterior_exact,
    get_log_posterior_exact_meanfield,
)


@pytest.fixture
def config():
    c = ExperimentConfig.default("test", "ising", "nbinom_cm")
    c.graph.set_value("size", 4)
    c.graph.edge_count.set_value("state", 4)
    c.graph.set_value("sample_graph_prior_prob", 0.0)
    c.graph.set_value("heterogeneity", 0.0)
    c.dynamics.set_value("num_steps", [1])
    c.dynamics.set_coupling(0.0)
    # c.dynamics.set_value("infection_prob", 0.5)
    # c.dynamics.set_value("recovery_prob", 0.5)
    c.insert("num_sweeps", 10000)
    c.insert("numsteps_between_samples", 5)
    return c


def mcmc_analysis(c, callbacks=None):
    g = RandomGraphFactory.build(c.graph)
    d = DynamicsFactory.build(c.dynamics)
    d.set_random_graph(g.get_wrap())
    g_mcmc = RandomGraphMCMCFactory.build(c.graph)
    mcmc = DynamicsMCMC(
        d, g_mcmc.get_wrap(), 1, 1, c.graph.sample_graph_prior_prob
    )
    d.sample()
    mcmc.set_up()
    measure = Frobenius().dist
    return Wrapper(
        MCMCConvergenceAnalysis(mcmc, measure, callbacks=callbacks),
        D_MCMC=mcmc,
        distance=measure,
        g_mcmc=g_mcmc,
        d=d,
        g=g,
    )


def logfac(n):
    return loggamma(n + 1)


def logdfac(n):
    k = n // 2
    return logfac(k) + np.log(2) * k


def test_meanfield(config):
    for c in config.sequence():
        callback = CollectEdgeMultiplicityOnSweep()
        conv = mcmc_analysis(c, [callback])
        entropy = []
        og_graph = conv.mcmc.get_graph()
        # hgx = -get_log_posterior_exact(conv.mcmc, None)
        # hxg = -conv.mcmc.get_log_likelihood()
        hg = -conv.mcmc.get_log_prior()
        # hx = hg + hxg - hgx
        hgx_mf = -get_log_posterior_exact_meanfield(conv.mcmc, None)
        # conv.burn(5000)
        print(conv.mcmc.get_random_graph_mcmc().get_edge_proposer())
        s, f = 0, 0
        print("Setting up")
        conv.mcmc.set_graph(og_graph)
        for n in range(c.num_sweeps):
            _s, _f = conv.burn(20)
            s += _s
            f += _f

            # entropy.append(callback.get_marginal_entropy())
            entropy.append(-callback.get_log_posterior_estimate(og_graph))
            # if (n % 10) == 0:
            #     print(n, s, f, entropy[-1] / -conv.mcmc.get_log_prior())
        # print(conv.mcmc.get_random_graph_mcmc().get_edge_proposer().get_edge_proposal_counts())
        # print(conv.mcmc.get_random_graph_mcmc().get_edge_proposer().get_vertex_proposal_counts())
        print(
            {
                e: n / (sum(conv.mcmc.get_added_edge_counter().values()))
                for e, n in conv.mcmc.get_added_edge_counter().items()
            }
        )
        print(
            {
                e: n / (sum(conv.mcmc.get_removed_edge_counter().values()))
                for e, n in conv.mcmc.get_removed_edge_counter().items()
            }
        )
        for e, w in callback.get_edge_probs().items():
            print("MCMC", e, w)
        plt.loglog(entropy, label=rf"$C = {c.dynamics.get_coupling()}$")
        plt.axhline(hg, color="grey", linestyle="--")

        # plt.axhline(hgx, color="grey", linestyle="-.")
        plt.axhline(hgx_mf, color="grey", linestyle="dotted")
        # print("Graph", og_graph)
        print("MCMC hgx", entropy[-1])
        print("hgx_mf", hgx_mf)
        print("hg", hg)
        # print("hgx", hgx)
        # print("hxg", hxg)
        # print("hx", hx)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    pass
