import numpy as np
import unittest

from midynet.experiments import Experiment
from midynet.config import ExperimentConfig
from midynet.util.loggers import MemoryLogger, TimeLogger


class TestExperimentBaseClass(unittest.TestCase):
    display: bool = False

    def setUp(self):
        self.config = ExperimentConfig.default(
            "test",
            "sis",
            "er",
            metrics=["dynamics_prediction_entropy", "graph_entropy"],
            path="./tests/experiments/test-dir",
            num_procs=4,
            seed=1,
        )
        coupling = np.linspace(0, 0.5, 2).tolist()
        self.config.set_value("dynamics.infection_prob", coupling)
        self.config.set_value("dynamics.recovery_prob", 0.5)
        self.config.set_value("dynamics.num_steps", 100)
        self.config.set_value("graph.size", 10)
        self.config.set_value("graph.edge_count.state", [0, 5])
        self.config.set_value("metrics.dynamics_prediction_entropy.num_samples", 5)

        self.exp = Experiment(
            self.config,
            verbose=0,
            loggers={"memory": MemoryLogger(), "time": TimeLogger()},
        )
        self.exp.begin()

    def tearDown(self):
        self.exp.path.rmdir()

    def test_config(self):
        if self.display:
            print(self.exp.config.format())

    def test_run(self):
        pass

    def test_begin(self):
        pass

    def test_end(self):
        pass

    def test_compute_metrics(self):
        self.exp.compute_metrics()
        for n in self.config.metrics.metrics_names:
            (self.exp.path / f"{n}.pickle").unlink()

    def test_save(self):
        pass

    def test_load(self):
        pass

    def test_load_from_file(self):
        pass

    def test_unzip(self):
        pass

    def test_clean(self):
        pass

    def test_merge(self):
        pass


if __name__ == "__main__":
    unittest.main()
