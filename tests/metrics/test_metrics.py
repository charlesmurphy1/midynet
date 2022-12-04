import pathlib
from dataclasses import dataclass

import numpy as np
import pytest

from midynet import metrics
from midynet.config import (
    ParameterSequence,
    MetaConfig,
    DataModelConfig,
    GraphConfig,
    MetricsCollectionConfig,
    ExperimentConfig,
)


class DummyMetrics(metrics.Metrics):
    def eval(self, config):
        return {"dummy": np.pi}


@pytest.fixture
def base_metrics():
    coupling = np.linspace(0, 10, 2)
    infection_prob = np.linspace(0, 10, 2)
    N = [10, 100]
    d = DataModelConfig.auto(["glauber", "sis"])
    d[0].coupling = ParameterSequence(coupling.tolist())
    d[1].infection_prob = ParameterSequence(infection_prob.tolist())

    g = GraphConfig.auto("erdosrenyi")
    g.size = N
    config = MetaConfig(name="test", data_model=d, prior=g)
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
        assert "dummy" in data
        assert (
            "data_model.coupling"
            if name == "test.glauber"
            else "data_model.infection_prob"
        ) in data
        assert np.all(data.dummy == np.pi)


# def test_basemetrics_save(base_metrics):
#     base_metrics.compute()
#     base_metrics.save("metrics.pickle")
#     pathlib.Path("metrics.pickle").unlink()


# def test_basemetrics_load(base_metrics):
#     base_metrics.compute()
#     base_metrics.save("metrics.pickle")
#     base_metrics.load("metrics.pickle")
#     pathlib.Path("metrics.pickle").unlink()


metrics_dict = {
    # "data_entropy": metrics.DataEntropyMetrics,
    # "data_prediction_entropy": metrics.DataPredictionEntropyMetrics,
    # "graph_entropy": metrics.GraphEntropyMetrics,
    # "graph_reconstruction_entropy": metrics.GraphReconstructionEntropyMetrics,
    # "reconstructability": metrics.ReconstructabilityMetrics,
    # "predictability": metrics.PredictabilityMetrics,
    "recon_information": metrics.ReconstructionInformationMeasuresMetrics,
    "heuristics": metrics.ReconstructionHeuristicsMetrics,
}


# @pytest.fixture(params=[k for k in metrics_dict.keys()])
# def args(request):
#     c = ExperimentConfig.reconstruction(
#         "test",
#         "sis",
#         "erdosrenyi",
#         path="./tests/experiments/test-dir",
#         num_procs=1,
#         seed=1,
#     )
#     c.set_value("data_model.num_steps", 100)
#     c.set_value("data_model.infection_prob", [0.0, 0.5])
#     c.set_value("prior.size", 5)
#     c.set_value("prior.edge_count", 5)
#     c.set_value(
#         "metrics",
#         MetricsCollectionConfig.auto(request.param),
#     )
#     mcf = c.metrics.get_value(request.param)
#     mcf.set_value("num_samples", 1)
#     if "method" in mcf and mcf.name != "heuristics":
#         mcf.set_value(
#             "method", ["arithmetic", "harmonic", "meanfield", "annealed"]
#         )
#         mcf.set_value("initial_burn", 1)
#         mcf.set_value("K", 2)
#         mcf.set_value("num_sweeps", 10)
#         mcf.set_value("burn_per_vertex", 1)
#         mcf.set_value("start_from_original", True)
#     elif mcf.name == "heuristics":
#         mcf.set_value(
#             "method",
#             [
#                 "correlation",
#                 "granger_causality",
#                 "transfer_entropy",
#                 "mutual_information",
#                 "partial_correlation",
#                 "correlation_spanning_tree",
#             ],
#         )
#     c.metrics.set_value(request.param, mcf)
#     return c, metrics_dict[request.param]


# def test_eval(args):
#     config, metrics = args
#     m = metrics(config)
#     for c in config.sequence():
#         m.eval(c)


if __name__ == "__main__":
    pass
