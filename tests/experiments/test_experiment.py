import numpy as np
import pytest

from midynet.config import ExperimentConfig
from midynet.experiments import Experiment
from midynet.util.loggers import MemoryLogger, TimeLogger

DISPLAY = False


@pytest.fixture
def experiment():
    config = ExperimentConfig.reconstruction(
        "test",
        "sis",
        "erdosrenyi",
        metrics=["data_prediction_entropy", "graph_entropy"],
        path="./testing/experiments/test-dir",
        num_procs=4,
        seed=1,
    )
    coupling = np.linspace(0, 0.5, 2).tolist()
    config.set_value("data_model.infection_prob", coupling)
    config.set_value("data_model.recovery_prob", 0.5)
    config.set_value("data_model.num_steps", 100)
    config.set_value("graph.size", 10)
    config.set_value("graph.edge_count.state", [0, 5])
    config.set_value("metrics.dynamics_prediction_entropy.num_samples", 5)

    exp = Experiment(
        config,
        verbose=0,
        loggers={"memory": MemoryLogger(), "time": TimeLogger()},
    )
    exp.begin()
    return exp


@pytest.fixture(autouse=True)
def tear_down(experiment):
    yield
    path = experiment.path
    path.rmdir()  # testing/experiments/test-dir
    path = path.parent
    path.rmdir()  # testing/experiments
    path = path.parent
    path.rmdir()  # testing


def test_config(experiment):
    if DISPLAY:
        print(experiment.config.format())


def test_run(experiment):
    pass


def test_begin(experiment):
    pass


def test_end(experiment):
    pass


def test_compute_metrics(experiment):
    experiment.compute_metrics()
    for n in experiment.config.metrics.metrics_names:
        (experiment.path / f"{n}.pickle").unlink()


def test_save(experiment):
    pass


def test_load(experiment):
    pass


def test_load_from_file(experiment):
    pass


def test_unzip(experiment):
    pass


def test_clean(experiment):
    pass


def test_merge(experiment):
    pass


if __name__ == "__main__":
    pass
