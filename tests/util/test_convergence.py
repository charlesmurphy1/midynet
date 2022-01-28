import unittest
import numpy as np
import matplotlib.pyplot as plt

from netrd.distance import Hamming
from midynet.config import *
from midynet.util import MCMCConvergenceAnalysis
from _midynet.mcmc import DynamicsMCMC


class TestMCMCConvergence(unittest.TestCase):
    compute: bool = True

    def setUp(self):
        self.config = ExperimentConfig.default("test", "cowan", "nbinom_cm")
        self.config.graph.set_value("size", 100)
        self.config.graph.edge_count.set_value("state", 250)
        self.config.graph.set_value("heterogeneity", 1.0)
        self.config.dynamics.set_value("num_steps", 1000)
        self.config.dynamics.set_coupling([0.0, 0.1, 0.2, 0.5, 0.9])
        self.num_samples = 100
        self.numsteps_between_samples = 25

    @staticmethod
    def setup_convergence(config):
        g = RandomGraphFactory.build(config.graph)
        d = DynamicsFactory.build(config.dynamics)
        d.set_random_graph(g.get_wrap())
        g_mcmc = RandomGraphMCMCFactory.build(config.graph)
        mcmc = DynamicsMCMC(d, g_mcmc.get_wrap())
        d.sample()
        d.sample_graph()
        mcmc.set_up()
        distance = Hamming()
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
        for c in self.config.sequence():
            conv = TestMCMCConvergence.setup_convergence(c)
            collected = conv.collect(self.num_samples, self.numsteps_between_samples)
            x = np.arange(self.num_samples) * self.numsteps_between_samples
            inf_prob = c.dynamics.nu
            plt.plot(x, collected, label=rf"$\alpha = {inf_prob}$")
        print(f"Computation time: {time.time() - t}")
        plt.xlabel("Number of MH steps")
        plt.ylabel("Hamming distance")
        plt.legend()
        plt.show()
        # plt.savefig(
        #     f"./tests/util/convergence_{self.config.dynamics.name}_{self.config.graph.name}.png"
        # )
