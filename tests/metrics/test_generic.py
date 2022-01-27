import unittest
import midynet
from midynet.config import *
import numpy as np


class TestGenericMetrics(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig.default(
            name="test", dynamics="ising", graph="ser"
        )
        self.config.dynamics.set_value("num_steps", 100)
        self.config.dynamics.set_value("coupling", 4)
        self.config.graph.set_value("size", 100)
        self.config.graph.edge_count.set_value("state", 250)

    def setup_mcmc(self):
        graph = RandomGraphFactory.build(self.config.graph)
        dynamics = DynamicsFactory.build(self.config.dynamics)
        dynamics.set_random_graph(graph.get_wrap())
        graph_mcmc = RandomGraphMCMCFactory.build(self.config.graph)
        mcmc = midynet.mcmc.DynamicsMCMC(dynamics, graph_mcmc.get_wrap())
        verbose = MCMCVerboseFactory.build("console")
        callback = midynet.mcmc.callbacks.CollectLikelihoodOnSweep()
        dynamics.sample()
        mcmc.set_up()
        return Wrapper(
            mcmc,
            dynamics=dynamics,
            graph=graph,
            graph_mcmc=graph_mcmc,
        )

    def test_generic(self):
        mcmc = self.setup_mcmc()
        metrics = MetricsConfig.mcmc()
        metrics.set_value("num_sweeps", 1000)
        # metrics.set_value("method", "exact")
        # hx = midynet.metrics.util.get_log_evidence(mcmc, metrics)
        # print(f"exact H(X): {hx}")

        metrics.set_value("method", "meanfield")
        hx = midynet.metrics.util.get_log_evidence(mcmc, metrics)
        print(f"meanfield H(X): {hx}")

        metrics.set_value("method", "annealed")
        metrics.set_value("num_betas", 20)
        metrics.set_value("exp_betas", 0.5)
        hx = midynet.metrics.util.get_log_evidence(mcmc, metrics)
        print(f"annealed H(X): {hx}")


if __name__ == "__main__":
    unittest.main()
