import midynet
import pytest

from midynet.config import *
from _midynet.mcmc import DynamicsMCMC

DISPLAY = False


@pytest.fixture
def config():
    c = ExperimentConfig.default(name="test", dynamics="sis", graph="er")
    c.dynamics.set_value("num_steps", 5)
    c.graph.set_value("size", 4)
    c.graph.edge_count.set_value("state", 2)
    return c


@pytest.fixture
def metrics_config():
    c = MetricsConfig.mcmc()
    c.set_value("num_sweeps", 10)
    c.set_value("burn_per_vertex", 1)
    return c


@pytest.fixture
def mcmc(config):
    graph = RandomGraphFactory.build(config.graph)
    dynamics = DynamicsFactory.build(config.dynamics)
    dynamics.set_random_graph(graph.get_wrap())
    graph_mcmc = RandomGraphMCMCFactory.build(config.graph)
    mcmc = DynamicsMCMC(dynamics, graph_mcmc.get_wrap(), 1, 1, 0)
    dynamics.sample()
    mcmc.set_up()
    return Wrapper(mcmc, dynamics=dynamics, graph=graph, graph_mcmc=graph_mcmc)


def test_log_evidence_arithmetic(mcmc, metrics_config):
    metrics_config.set_value("method", "arithmetic")
    logp = midynet.metrics.util.get_log_evidence(mcmc, metrics_config)
    if DISPLAY:
        print("arithmetic", logp)


def test_log_evidence_harmonic(mcmc, metrics_config):
    metrics_config.set_value("method", "harmonic")
    logp = midynet.metrics.util.get_log_evidence(mcmc, metrics_config)

    if DISPLAY:
        print("harmonic", logp)


def test_log_evidence_annealed(mcmc, metrics_config):
    metrics_config.set_value("method", "annealed")
    logp = midynet.metrics.util.get_log_evidence(mcmc, metrics_config)

    if DISPLAY:
        print("annealed", logp)


def test_log_evidence_meanfield(mcmc, metrics_config):
    metrics_config.set_value("method", "meanfield")
    logp = midynet.metrics.util.get_log_evidence(mcmc, metrics_config)

    if DISPLAY:
        print("meanfield", logp)


def test_log_evidence_exact(mcmc, metrics_config):
    metrics_config.set_value("method", "exact")
    logp = midynet.metrics.util.get_log_evidence(mcmc, metrics_config)

    if DISPLAY:
        print("exact", logp)


def test_log_evidence_exact_meanfield(mcmc, metrics_config):
    metrics_config.set_value("method", "exact_meanfield")
    logp = midynet.metrics.util.get_log_evidence(mcmc, metrics_config)

    if DISPLAY:
        print("exact_meanfield", logp)


if __name__ == "__main__":
    pass
