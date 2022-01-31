import unittest
import numpy as np
import matplotlib.pyplot as plt

from netrd.distance import Hamming, Frobenius
from midynet.config import *
from midynet.util import MCMCConvergenceAnalysis
from _midynet.mcmc import DynamicsMCMC


class TestMCMCConvergence(unittest.TestCase):
    compute: bool = True

    def setUp(self):
        self.config = ExperimentConfig.default("test", "ising", "nbinom_cm")
        self.config.graph.set_value("size", 50)
        self.config.graph.edge_count.set_value("state", 100)
        self.config.graph.set_value("sample_graph_prior_prob", 0.0)
        self.config.graph.set_value("heterogeneity", 0.001)
        self.config.dynamics.set_value("num_steps", 1000)
        self.config.dynamics.set_coupling([0.0, 0.5])
        self.num_samples = 1000
        self.numsteps_between_samples = 5
        print(self.config.format())

    @staticmethod
    def setup_convergence(config):
        g = RandomGraphFactory.build(config.graph)
        d = DynamicsFactory.build(config.dynamics)
        d.set_random_graph(g.get_wrap())
        g_mcmc = RandomGraphMCMCFactory.build(config.graph)
        mcmc = DynamicsMCMC(
            d, g_mcmc.get_wrap(), 1, 1, config.graph.sample_graph_prior_prob
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

    def test_generic(self):
        if not self.compute:
            return
        import time

        t = time.time()
        distance = None
        for c in self.config.sequence():
            conv = TestMCMCConvergence.setup_convergence(c)
            collected = conv.collect(
                burn=500,
                num_samples=self.num_samples,
                numsteps_between_samples=self.numsteps_between_samples,
            )
            x = np.arange(self.num_samples) * self.numsteps_between_samples
            inf_prob = c.dynamics.coupling
            plt.plot(x, collected, label=rf"$\alpha = {inf_prob}$")
            distance = conv.get_other("distance") if distance is None else distance
        print(f"Computation time: {time.time() - t}")
        plt.xlabel("Number of MH steps")
        plt.ylabel(f"{distance.__class__.__name__} distance")
        plt.legend()
        plt.show()
        # plt.savefig(
        #     f"./tests/util/convergence_{self.config.dynamics.name}_{self.config.graph.name}.png"
        # )
