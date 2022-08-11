import pytest
import midynet
from midynet.config import (
    DataModelFactory,
    RandomGraphFactory,
    MetricsConfig,
    ExperimentConfig,
    Wrapper,
    ReconstructionMCMC,
)

DISPLAY = False


@pytest.fixture
def config():
    c = ExperimentConfig.reconstruction(
        name="test", data_model="sis", graph="erdosrenyi"
    )
    c.data_model.set_value("num_steps", 5)
    c.graph.set_value("size", 4)
    c.graph.set_value("edge_count", 2)
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
    dynamics = DataModelFactory.build(config.data_model)
    dynamics.set_graph_prior(graph)
    print(dynamics)
    mcmc = ReconstructionMCMC(dynamics, graph)
    mcmc.sample()
    return mcmc


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


def test_log_prior_meanfield(mcmc, metrics_config):
    logp = midynet.metrics.util.get_log_prior_meanfield(mcmc, metrics_config)

    if DISPLAY:
        print("prior-meanfield", logp)


if __name__ == "__main__":
    pass
