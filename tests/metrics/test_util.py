import unittest

import midynet

from midynet.config import *
from _midynet.mcmc import DynamicsMCMC


class TestMetricsUtil(unittest.TestCase):
    display: bool = False

    def setUp(self):
        self.config = midynet.config.ExperimentConfig.default(
            name="test", dynamics="sis", graph="er"
        )
        self.config.dynamics.set_value("num_steps", 5)
        self.config.graph.set_value("size", 4)
        self.config.graph.edge_count.set_value("state", 2)
        self.metrics_config = midynet.config.MetricsConfig.mcmc()
        self.metrics_config.set_value("num_sweeps", 100)
        self.metrics_config.set_value("burn_per_vertex", 10)

    def setup_mcmc(self):
        graph = RandomGraphFactory.build(self.config.graph)
        dynamics = DynamicsFactory.build(self.config.dynamics)
        dynamics.set_random_graph(graph.get_wrap())
        graph_mcmc = RandomGraphMCMCFactory.build(self.config.graph)
        mcmc = DynamicsMCMC(dynamics, graph_mcmc.get_wrap())
        mcmc.sample()
        mcmc.set_up()
        return midynet.config.Wrapper(
            mcmc, dynamics=dynamics, graph=graph, graph_mcmc=graph_mcmc
        )

    def test_log_evidence_arithmetic(self):
        mcmc = self.setup_mcmc()
        self.metrics_config.set_value("method", "arithmetic")
        logp = midynet.metrics.util.get_log_evidence(mcmc, self.metrics_config)
        if self.display:
            print("arithmetic", logp)

    def test_log_evidence_harmonic(self):
        mcmc = self.setup_mcmc()
        self.metrics_config.set_value("method", "harmonic")
        logp = midynet.metrics.util.get_log_evidence(mcmc, self.metrics_config)

        if self.display:
            print("harmonic", logp)

    def test_log_evidence_annealed(self):
        mcmc = self.setup_mcmc()
        self.metrics_config.set_value("method", "annealed")
        logp = midynet.metrics.util.get_log_evidence(mcmc, self.metrics_config)

        if self.display:
            print("annealed", logp)

    def test_log_evidence_meanfield(self):
        mcmc = self.setup_mcmc()
        self.metrics_config.set_value("method", "meanfield")
        logp = midynet.metrics.util.get_log_evidence(mcmc, self.metrics_config)

        if self.display:
            print("meanfield", logp)

    def test_log_evidence_exact(self):
        mcmc = self.setup_mcmc()
        self.metrics_config.set_value("method", "exact")
        logp = midynet.metrics.util.get_log_evidence(mcmc, self.metrics_config)

        if self.display:
            print("exact", logp)

    def test_log_evidence_exact_meanfield(self):
        mcmc = self.setup_mcmc()
        self.metrics_config.set_value("method", "exact_meanfield")
        logp = midynet.metrics.util.get_log_evidence(mcmc, self.metrics_config)

        if self.display:
            print("exact_meanfield", logp)


if __name__ == "__main__":
    unittest.main()
