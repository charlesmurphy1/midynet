import itertools
import numpy as np
import pathlib
import time
import unittest
from dataclasses import dataclass, field

import midynet
from midynet.config import *
from midynet import metrics


@dataclass
class DummyMetrics(metrics.Metrics):
    value: float = np.pi

    def eval(self, config):
        return {"dummy": self.value}


@dataclass
class DummyExperiment:
    config: Config = field(repr=False, default_factory=Config)
    num_procs: int = 1


class TestMetricsBaseClass(unittest.TestCase):
    time_it: bool = False

    def setUp(self):
        if self.time_it:
            self.begin = time.time()
        self.coupling = np.linspace(0, 10, 2)
        self.infection_prob = np.linspace(0, 10, 2)
        self.N = [10, 100]
        self.coupling_size = len(self.coupling)
        self.N_size = len(self.N)
        d = DynamicsConfig.auto(["ising", "sis"])
        d[0].set_value("coupling", self.coupling)
        d[1].set_value("infection_prob", self.infection_prob)

        g = RandomGraphConfig.auto("uniform_sbm")
        g.set_value("size", self.N)
        self.config = Config(name="test", dynamics=d, graph=g)
        self.experiment = DummyExperiment(config=self.config)
        self.metrics = DummyMetrics(config=self.config)

    def tearDown(self):
        if self.time_it:
            print()
            print(f"Time: {time.time() - self.begin}")

    def test_set_up(self):
        pass

    def test_tear_down(self):
        pass

    def test_eval(self):
        pass

    def test_compute(self):
        self.metrics.compute()
        for name, data in self.metrics.data.items():
            for key, value in data.items():
                if name == "test.ising":
                    self.assertEqual(value.shape, (len(self.coupling), len(self.N)))
                elif name == "test.sis":
                    self.assertEqual(
                        value.shape, (len(self.infection_prob), len(self.N))
                    )
                self.assertTrue(np.all(value == np.pi))

    def test_format_data(self):
        self.metrics.compute()

    def test_flatten(self):
        self.metrics.compute()
        flat_data = self.metrics.flatten(self.metrics.data)
        for name, data in flat_data.items():
            for key, value in data.items():
                if name == "test.ising":
                    self.assertEqual(value.shape, (len(self.coupling) * len(self.N),))
                elif name == "test.sis":
                    self.assertEqual(
                        value.shape, (len(self.infection_prob) * len(self.N),)
                    )
                self.assertTrue(np.all(value == np.pi))

    def test_save(self):
        self.metrics.compute()
        self.metrics.save("metrics.pickle")
        pathlib.Path("metrics.pickle").unlink()

    def test_load(self):
        self.metrics.compute()
        self.metrics.save("metrics.pickle")
        self.metrics.load("metrics.pickle")
        pathlib.Path("metrics.pickle").unlink()

    def test_merge_with(self):
        config = self.config.deepcopy()

        config.set_value("dynamics.ising.coupling", 1000.0)
        config.set_value("dynamics.sis.infection_prob", 1000.0)
        config.set_value(
            "graph.edge_matrix.edge_count", EdgeCountPriorConfig.auto(50.0)
        )
        metrics = DummyMetrics(value=2, config=config)
        metrics.compute()
        self.metrics.compute()
        self.metrics.merge_with(metrics)

        d = self.metrics.data
        for keys in itertools.product(["test"], ["ising", "sis"], ["delta", "poisson"]):
            name = ".".join(keys)
            self.assertIn(name, d)
            self.assertEqual(
                d[name]["dummy"].shape, (self.coupling_size + 1, self.N_size)
            )


class TemplateTestMetrics:
    _metrics: metrics.Metrics
    display: bool = False
    not_implemented: bool = False

    def setUp(self):
        self.config = ExperimentConfig.default(
            "test",
            "sis",
            "er",
            path="./tests/experiments/test-dir",
            num_procs=1,
            seed=1,
        )
        self.config.set_value("dynamics.num_steps", 10)
        self.config.set_value("dynamics.infection_prob", [0.0, 0.5])
        self.config.set_value("graph.size", 10)
        self.config.set_value("graph.edge_count.state", 25)
        self.metrics = self._metrics(self.config)

    def test_eval(self):
        if self.not_implemented:
            with self.assertRaises(NotImplementedError):
                data = self.metrics.eval(self.config)
        else:
            for c in self.config.sequence():
                data = self.metrics.eval(c)
                if self.display:
                    print(c.format())
                    print(data)


class TestDynamicsEntropy(unittest.TestCase, TemplateTestMetrics):
    _metrics: metrics.Metrics = metrics.DynamicsEntropyMetrics

    def setUp(self):
        TemplateTestMetrics.setUp(self)
        self.config.set_value(
            "metrics", MetricsCollectionConfig.auto("dynamics_entropy")
        )
        self.config.metrics.dynamics_entropy.set_value("num_samples", 5)
        self.config.metrics.dynamics_entropy.set_value(
            "method", ["arithmetic", "harmonic", "meanfield", "annealed"]
        )
        self.config.metrics.dynamics_entropy.set_value("K", 2)
        self.config.metrics.dynamics_entropy.set_value("num_sweeps", 10)


class TestDynamicsPredictionEntropy(unittest.TestCase, TemplateTestMetrics):
    _metrics: metrics.Metrics = metrics.DynamicsPredictionEntropyMetrics

    def setUp(self):
        TemplateTestMetrics.setUp(self)
        self.config.set_value(
            "metrics", MetricsCollectionConfig.auto("dynamics_prediction_entropy")
        )
        self.config.metrics.dynamics_prediction_entropy.set_value("num_samples", 24)


class TestGraphEntropy(unittest.TestCase, TemplateTestMetrics):
    _metrics: metrics.Metrics = metrics.GraphEntropyMetrics

    def setUp(self):
        TemplateTestMetrics.setUp(self)
        self.config.set_value("metrics", MetricsCollectionConfig.auto("graph_entropy"))
        self.config.metrics.graph_entropy.set_value("num_samples", 24)


class TestGraphReconstructionEntropy(unittest.TestCase, TemplateTestMetrics):
    _metrics: metrics.Metrics = metrics.GraphReconstructionEntropyMetrics
    not_implemented: bool = False

    def setUp(self):
        TemplateTestMetrics.setUp(self)
        self.config.set_value(
            "metrics", MetricsCollectionConfig.auto("graph_reconstruction_entropy")
        )
        self.config.metrics.graph_reconstruction_entropy.set_value("num_samples", 5)
        self.config.metrics.graph_reconstruction_entropy.set_value(
            "method", ["arithmetic", "harmonic", "meanfield", "annealed"]
        )
        self.config.metrics.graph_reconstruction_entropy.set_value("K", 2)
        self.config.metrics.graph_reconstruction_entropy.set_value("num_sweeps", 10)


class TestReconstructability(unittest.TestCase, TemplateTestMetrics):
    _metrics: metrics.Metrics = metrics.ReconstructabilityMetrics

    def setUp(self):
        TemplateTestMetrics.setUp(self)
        self.config.set_value(
            "metrics", MetricsCollectionConfig.auto("reconstructability")
        )
        self.config.metrics.reconstructability.set_value("num_samples", 5)
        self.config.metrics.reconstructability.set_value(
            "method", ["arithmetic", "harmonic", "meanfield", "annealed"]
        )
        self.config.metrics.reconstructability.set_value("K", 2)
        self.config.metrics.reconstructability.set_value("num_sweeps", 10)


class TestPredictability(unittest.TestCase, TemplateTestMetrics):
    _metrics: metrics.Metrics = metrics.PredictabilityMetrics

    def setUp(self):
        TemplateTestMetrics.setUp(self)
        self.config.set_value("metrics", MetricsCollectionConfig.auto("predictability"))
        self.config.metrics.predictability.set_value("num_samples", 5)
        self.config.metrics.predictability.set_value(
            "method", ["arithmetic", "harmonic", "meanfield", "annealed"]
        )
        self.config.metrics.predictability.set_value("K", 2)
        self.config.metrics.predictability.set_value("num_sweeps", 10)


class TestMutualInformation(unittest.TestCase, TemplateTestMetrics):
    _metrics: metrics.Metrics = metrics.MutualInformationMetrics

    def setUp(self):
        TemplateTestMetrics.setUp(self)
        self.config.set_value("metrics", MetricsCollectionConfig.auto("mutualinfo"))
        self.config.metrics.mutualinfo.set_value("num_samples", 5)
        self.config.metrics.mutualinfo.set_value(
            "method", ["arithmetic", "harmonic", "meanfield", "annealed"]
        )
        self.config.metrics.mutualinfo.set_value("K", 2)
        self.config.metrics.mutualinfo.set_value("num_sweeps", 10)


if __name__ == "__main__":
    unittest.main()
