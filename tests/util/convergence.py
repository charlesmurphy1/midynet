import time
import matplotlib.pyplot as plt
import numpy as np
import pytest
from _midynet.mcmc import DynamicsMCMC
from _midynet.mcmc.callbacks import CollectEdgeMultiplicityOnSweep
from netrd.distance import Frobenius

from midynet.config import (
    DynamicsFactory,
    RandomGraphFactory,
    RandomGraphMCMCFactory,
    ExperimentConfig,
    Wrapper,
)
from midynet.util import MCMCConvergenceAnalysis


@pytest.fixture
def config():
    c = ExperimentConfig.default("test", "ising", "nbinom_cm")
    c.graph.set_value("size", 100)
    c.graph.edge_count.set_value("state", 250)
    c.graph.set_value("sample_graph_prior_prob", 0.0)
    c.graph.set_value("heterogeneity", 1.)
    c.dynamics.set_value("num_steps", [100])
    c.dynamics.set_coupling([1.0])
    c.insert("num_sweeps", 1000)
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
    d.sample_graph()
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


# def test_generic(config):
#
#     t = time.time()
#     distance = None
#     for c in config.sequence():
#         conv = mcmc_analysis(c)
#         collected = conv.collect(
#             burn=500,
#             num_sweeps=c.num_sweeps,
#             numsteps_between_samples=c.numsteps_between_samples,
#         )
#         x = np.arange(c.num_sweeps) * c.numsteps_between_samples
#         coupling = c.dynamics.coupling
#         plt.plot(x, collected, label=rf"$\alpha = {coupling}$")
#         distance = conv.get_other("distance") if distance is None else distance
#     print(f"Computation time: {time.time() - t}")
#     plt.xlabel("Number of MH steps")
#     plt.ylabel(f"{distance.__class__.__name__} distance")
#     plt.legend()
#     plt.show()


def test_meanfield(config):
    for c in config.sequence():
        callback = CollectEdgeMultiplicityOnSweep()
        conv = mcmc_analysis(c, [callback])
        entropy = []
        conv.burn(1000)
        for n in range(c.num_sweeps):
            if (n % 10) == 0:
                print(n)
            conv.burn(250)
            entropy.append(callback.get_marginal_entropy())

        plt.loglog(entropy, label=rf"$C = {c.dynamics.num_steps}$")
        plt.axhline(-conv.mcmc.get_log_prior(), color="grey", linestyle="--")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    pass
