import time

import matplotlib.pyplot as plt
import numpy as np
import pytest
from _midynet.mcmc import DynamicsMCMC
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
    c.graph.set_value("size", 50)
    c.graph.edge_count.set_value("state", 100)
    c.graph.set_value("sample_graph_prior_prob", 0.0)
    c.graph.set_value("heterogeneity", 0.001)
    c.dynamics.set_value("num_steps", 1000)
    c.dynamics.set_coupling([0.0, 0.5])
    c.insert("num_samples", 1000)
    c.insert("numsteps_between_samples", 5)
    return c


def mcmc_analysis(c):
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
    distance = Frobenius()
    return Wrapper(
        MCMCConvergenceAnalysis(mcmc, distance),
        D_MCMC=mcmc,
        distance=distance,
        g_mcmc=g_mcmc,
        d=d,
        g=g,
    )


def test_generic(config):

    t = time.time()
    distance = None
    for c in config.sequence():
        conv = mcmc_analysis(c)
        collected = conv.collect(
            burn=500,
            num_samples=c.num_samples,
            numsteps_between_samples=c.numsteps_between_samples,
        )
        x = np.arange(c.num_samples) * c.numsteps_between_samples
        coupling = c.dynamics.coupling
        plt.plot(x, collected, label=rf"$\alpha = {coupling}$")
        distance = conv.get_other("distance") if distance is None else distance
    print(f"Computation time: {time.time() - t}")
    plt.xlabel("Number of MH steps")
    plt.ylabel(f"{distance.__class__.__name__} distance")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    pass
