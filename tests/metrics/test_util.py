import pytest
import midynet
from midynet.config import (
    DataModelFactory,
    GraphFactory,
    MetricsConfig,
    ExperimentConfig,
)
from graphinf.mcmc import GraphReconstructionMCMC

DISPLAY = False


@pytest.fixture
def config():
    c = ExperimentConfig.reconstruction(
        name="test", data_model="sis", prior="erdosrenyi"
    )
    c.data_model.length = 5
    c.prior.size = 4
    c.prior.edge_count = 2
    return c


@pytest.fixture
def metrics_config():
    c = MetricsConfig.mcmc("test")
    c.num_sweeps = 10
    c.burn_per_vertex = 1
    return c


@pytest.fixture
def data_model(config):
    prior = GraphFactory.build(config.prior)
    data_model = DataModelFactory.build(config.data_model)
    data_model.set_graph_prior(prior)
    data_model.sample()
    return data_model


def test_log_evidence_arithmetic(data_model, metrics_config):
    metrics_config.method = "arithmetic"
    logp = midynet.metrics.util.get_log_evidence(data_model, metrics_config)
    if DISPLAY:
        print("arithmetic", logp)


def test_log_evidence_harmonic(data_model, metrics_config):
    metrics_config.method = "harmonic"
    logp = midynet.metrics.util.get_log_evidence(data_model, metrics_config)

    if DISPLAY:
        print("harmonic", logp)


def test_log_evidence_annealed(data_model, metrics_config):
    metrics_config.method = "annealed"
    logp = midynet.metrics.util.get_log_evidence(data_model, metrics_config)

    if DISPLAY:
        print("annealed", logp)


def test_log_evidence_meanfield(data_model, metrics_config):
    metrics_config.method = "meanfield"
    logp = midynet.metrics.util.get_log_evidence(data_model, metrics_config)

    if DISPLAY:
        print("meanfield", logp)


def test_log_evidence_exact(data_model, metrics_config):
    metrics_config.method = "exact"
    logp = midynet.metrics.util.get_log_evidence(data_model, metrics_config)

    if DISPLAY:
        print("exact", logp)


def test_log_evidence_exact_meanfield(data_model, metrics_config):
    metrics_config.method = "exact_meanfield"
    logp = midynet.metrics.util.get_log_evidence(data_model, metrics_config)

    if DISPLAY:
        print("exact_meanfield", logp)


def test_log_prior_meanfield(data_model, metrics_config):
    logp = midynet.metrics.util.get_log_prior_meanfield(data_model, metrics_config)

    if DISPLAY:
        print("prior-meanfield", logp)


if __name__ == "__main__":
    pass
