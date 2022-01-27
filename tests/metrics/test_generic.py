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
        self.config.graph.set_value("size", 5)
        self.config.graph.edge_count.set_value("state", 5)

    def setup_mcmc(self):
        graph = RandomGraphFactory.build(self.config.graph)
        dynamics = DynamicsFactory.build(self.config.dynamics)
        dynamics.set_random_graph(graph.get_wrap())
        graph_mcmc = RandomGraphMCMCFactory.build(self.config.graph)
        mcmc = midynet.mcmc.DynamicsMCMC(dynamics, graph_mcmc.get_wrap())
        verbose = MCMCVerboseFactory.build("console")
        callback = midynet.mcmc.callbacks.CollectLikelihoodOnSweep()
        # mcmc.add_callback(verbose.get_wrap())
        # mcmc.add_callback(callback)
        dynamics.sample()
        mcmc.set_up()
        return Wrapper(
            mcmc,
            dynamics=dynamics,
            graph=graph,
            graph_mcmc=graph_mcmc,
            # verbose=verbose,
            # callback=callback,
        )

    def test_generic(self):
        mcmc = self.setup_mcmc()
        metrics = MetricsConfig.mcmc()
        metrics.set_value("method", "exact")
        print(metrics.format())
        hx = midynet.metrics.util.get_log_evidence(mcmc, metrics)
        print(hx)

        metrics.set_value("method", "annealed")
        metrics.set_value("num_betas", 20)
        metrics.set_value("exp_betas", 0.5)
        print(metrics.format())
        hx = midynet.metrics.util.get_log_evidence(mcmc, metrics)
        print(hx)


if __name__ == "__main__":
    unittest.main()
