import pathlib
from dataclasses import dataclass

import numpy as np
import pytest

import midynet
from midynet import metrics
from midynet.config import *


@dataclass
class DummyMetrics(metrics.Metrics):
    value: float = np.pi

    def eval(self, config):
        return {"dummy": self.value}


@pytest.fixture
def base_metrics():
    coupling = np.linspace(0, 10, 2)
    infection_prob = np.linspace(0, 10, 2)
    N = [10, 100]
    coupling_size = len(coupling)
    N_size = len(N)
    d = DynamicsConfig.auto(["ising", "sis"])
    d[0].set_value("coupling", coupling)
    d[1].set_value("infection_prob", infection_prob)

    g = RandomGraphConfig.auto("uniform_sbm")
    g.set_value("size", N)
    config = Config(name="test", dynamics=d, graph=g)
    return DummyMetrics(config=config)


def test_basemetrics_set_up():
    pass


def test_basemetrics_tear_down():
    pass


def test_basemetrics_eval():
    pass


def test_basemetrics_compute(base_metrics):
    base_metrics.compute()
    for name, data in base_metrics.data.items():
        for key, value in data.items():
            # if name == "test.ising":
            #     assert value.shape == (len(self.coupling), len(self.N))
            # elif name == "test.sis":
            #     assert value.shape == (len(self.infection_prob), len(self.N))
            assert np.all(value == np.pi)


def test_basemetrics_format_data(base_metrics):
    base_metrics.compute()


def test_basemetrics_flatten(base_metrics):
    base_metrics.compute()
    flat_data = base_metrics.flatten(base_metrics.data)
    for name, data in flat_data.items():
        for key, value in data.items():
            # if name == "test.ising":
            #     assert value.shape == (len(self.coupling) * len(self.N),)
            # elif name == "test.sis":
            #     assert value.shape == (len(self.infection_prob) * len(self.N),)
            assert np.all(value == np.pi)


def test_basemetrics_save(base_metrics):
    base_metrics.compute()
    base_metrics.save("metrics.pickle")
    pathlib.Path("metrics.pickle").unlink()


def test_basemetrics_load(base_metrics):
    base_metrics.compute()
    base_metrics.save("metrics.pickle")
    base_metrics.load("metrics.pickle")
    pathlib.Path("metrics.pickle").unlink()


# def test_merge_with():
#     config = self.config.deepcopy()
#
#     config.set_value("dynamics.ising.coupling", 1000.0)
#     config.set_value("dynamics.sis.infection_prob", 1000.0)
#     config.set_value(
#         "graph.edge_matrix.edge_count", EdgeCountPriorConfig.auto(50.0)
#     )
#     metrics = DummyMetrics(value=2, config=config)
#     metrics.compute()
#     self.metrics.compute()
#     self.metrics.merge_with(metrics)
#
#     d = self.metrics.data
#     for keys in itertools.product(["test"], ["ising", "sis"], ["delta", "poisson"]):
#         name = ".".join(keys)
#         self.assertIn(name, d)
#         self.assertEqual(
#             d[name]["dummy"].shape, (self.coupling_size + 1, self.N_size)
#         )


metrics_dict = {
    "dynamics_entropy": metrics.DynamicsEntropyMetrics,
    "dynamics_prediction_entropy": metrics.DynamicsPredictionEntropyMetrics,
    "graph_entropy": metrics.GraphEntropyMetrics,
    "graph_reconstruction_entropy": metrics.GraphReconstructionEntropyMetrics,
    "reconstructability": metrics.ReconstructabilityMetrics,
    "predictability": metrics.PredictabilityMetrics,
    "mutualinfo": metrics.MutualInformationMetrics,
}


@pytest.fixture(params=[k for k in metrics_dict.keys()])
def args(request):
    c = ExperimentConfig.default(
        "test",
        "sis",
        "er",
        path="./tests/experiments/test-dir",
        num_procs=1,
        seed=1,
    )
    c.set_value("dynamics.num_steps", 10)
    c.set_value("dynamics.infection_prob", [0.0, 0.5])
    c.set_value("graph.size", 10)
    c.set_value("graph.edge_count.state", 25)
    c.set_value(
        "metrics",
        MetricsCollectionConfig.auto(request.param),
    )
    c.metrics.get_value(request.param).set_value("num_samples", 1)
    if "method" in c.metrics.get_value(request.param):
        c.metrics.get_value(request.param).set_value(
            "method", ["arithmetic", "harmonic", "meanfield", "annealed"]
        )
        c.metrics.get_value(request.param).set_value("initial_burn", 1)
        c.metrics.get_value(request.param).set_value("K", 2)
        c.metrics.get_value(request.param).set_value("num_sweeps", 10)
        c.metrics.get_value(request.param).set_value("burn_per_vertex", 1)
    return c, metrics_dict[request.param]


def test_eval(args):
    config, metrics = args
    m = metrics(config)
    for c in config.sequence():
        d = m.eval(c)


if __name__ == "__main__":
    pass
