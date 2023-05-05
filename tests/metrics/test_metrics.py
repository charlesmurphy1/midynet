import pathlib

import numpy as np
import pytest

from midynet import metrics
from midynet.config import (
    Config,
    DataModelConfig,
    GraphConfig,
    MetricsCollectionConfig,
    ExperimentConfig,
)


class DummyMetrics(metrics.Metrics):
    def eval(self, config):
        # time.sleep(0.1)
        return {"dummy": np.pi}


class DummyLogger:
    def __init__(self, name="Dummy"):
        self.name = name

    def info(self, msg):
        print(msg)


@pytest.fixture
def config():
    coupling = np.linspace(0, 10, 2)
    infection_prob = np.linspace(0, 10, 2)
    N = [10, 100]
    d = DataModelConfig.auto(["glauber", "sis"])
    d[0].coupling = coupling.tolist()
    d[1].infection_prob = infection_prob.tolist()

    g = GraphConfig.auto("erdosrenyi")
    g.size = N
    config = Config(name="test", data_model=d, prior=g, path="./")
    return config


@pytest.fixture
def basemetrics():
    return DummyMetrics()


def test_basemetrics_set_up():
    pass


def test_basemetrics_tear_down():
    pass


def test_basemetrics_eval():
    pass


def test_basemetrics_compute(config, basemetrics):
    basemetrics.compute(config)
    for name, data in basemetrics.data.items():
        assert "dummy" in data["metrics"]
        assert (
            "data_model.coupling"
            if name == "test.glauber"
            else "data_model.infection_prob"
        ) in data["params"]
        assert np.all(data["metrics"].dummy == np.pi)


def test_basemetrics_to_pickle(config, basemetrics):
    basemetrics.compute(config)
    basemetrics.to_pickle(config.path)
    pathlib.Path("./metrics.pkl").unlink()


def test_basemetrics_read_pickle(config, basemetrics):
    basemetrics.compute(config)
    basemetrics.to_pickle(config.path)
    basemetrics.read_pickle("./metrics.pkl")
    pathlib.Path("./metrics.pkl").unlink()


metrics_dict = {
    "reconinfo": metrics.ReconstructionInformationMeasuresMetrics,
    "reconheuristics": metrics.ReconstructionHeuristicsMetrics,
}


@pytest.fixture(params=[k for k in metrics_dict.keys()])
def args(request):
    c = ExperimentConfig.default(
        "test",
        "sis",
        "erdosrenyi",
        path="./tests/experiments/test-dir",
        num_workers=1,
        seed=1,
    )
    c.data_model.length = 5
    c.data_model.infection_prob = [0.0, 0.5]
    c.data_model.as_sequence("infection_prob")
    c.prior.size = 3
    c.prior.edge_count = 2
    c.metrics = MetricsCollectionConfig.auto(request.param)
    name = c.metrics.metrics_names[0]
    if name == "reconinfo":
        c.metrics.reconinfo.method = [
            "arithmetic",
            "harmonic",
            "meanfield",
            "annealed",
        ]
        c.metrics.reconinfo.as_sequence("method")
        c.metrics.reconinfo.initial_burn = 1
        c.metrics.reconinfo.K = 2
        c.metrics.reconinfo.num_sweeps = 10
        c.metrics.reconinfo.burn_per_vertex = 1
        c.metrics.reconinfo.start_from_original = True
        c.metrics.reconinfo.num_samples = 1
    elif name == "reconheuristics":
        c.metrics.reconheuristics.method = [
            "correlation",
            "granger_causality",
            "transfer_entropy",
            "mutual_information",
            "partial_correlation",
            "correlation_spanning_tree",
        ]
        c.metrics.reconheuristics.as_sequence("method")
        c.metrics.reconheuristics.num_samples = 1
    return c, metrics_dict[request.param]


def test_eval(args):
    config, metrics = args
    
    m = metrics()
    for c in config.to_sequence():
        c = c.copy()
        c.metrics = c.metrics[metrics.shortname]
        m.eval(c)


if __name__ == "__main__":
    pass
