import numpy as np
import unittest
import pathlib
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
    def setUp(self):
        d = DynamicsConfig.auto(["ising", "sis"])
        d[0].set_value("coupling", [1, 2, 3, 4, 5])
        d[1].set_value("infection_prob", [1, 4, 5])

        g = RandomGraphConfig.auto("uniform_sbm")
        g.set_value("size", [10, 25, 50, 100])
        self.config = Config(name="test", dynamics=d, graph=g)
        self.experiment = DummyExperiment(config=self.config)
        self.metrics = DummyMetrics(config=self.config)

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
                    self.assertEqual(value.shape, (5, 4))
                elif name == "test.sis":
                    self.assertEqual(value.shape, (3, 4))
                self.assertTrue(np.all(value == np.pi))

    def test_format_data(self):
        self.metrics.compute()

    def test_flatten(self):
        self.metrics.compute()
        flat_data = self.metrics.flatten(self.metrics.data)
        for name, data in flat_data.items():
            for key, value in data.items():
                if name == "test.ising":
                    self.assertEqual(value.shape, (5 * 4,))
                elif name == "test.sis":
                    self.assertEqual(value.shape, (3 * 4,))
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

    def test_merge(self):
        config = self.config.deepcopy()

        config.set_value("dynamics.ising.coupling", 6)
        config.set_value("dynamics.sis.infection_prob", 6)
        metrics = DummyMetrics(value=2, config=config)
        self.metrics.compute()
        metrics.compute()
        print(self.metrics.data)
        print(metrics.data)
        self.metrics.merge(metrics)


class TemplateTestMetrics:
    _metrics: metrics.Metrics
    graph_config: RandomGraphConfig = RandomGraphConfig.auto("hyperuniform_sbm")
    dynamics_config: DynamicsConfig = DynamicsConfig.auto("sis")
    metrics_config: Config = Config(name="metrics", num_procs=4, num_samples=10)
    config: Config = Config(
        name="test",
        graph=graph_config,
        dynamics=dynamics_config,
        metrics=metrics_config,
    )
    display: bool = False
    not_implemented: bool = False

    def test_eval(self):
        if self.not_implemented:
            with self.assertRaises(NotImplementedError):
                data = self._metrics.eval(self.config)
        else:
            data = self._metrics.eval(self.config)
            if self.display:
                print(data)


class TestDynamicsEntropy(unittest.TestCase, TemplateTestMetrics):
    _metrics: metrics.Metrics = metrics.DynamicsEntropyMetrics()
    not_implemented: bool = True


class TestDynamicsPredictionEntropy(unittest.TestCase, TemplateTestMetrics):
    _metrics: metrics.Metrics = metrics.DynamicsPredictionEntropyMetrics()


class TestGraphEntropy(unittest.TestCase, TemplateTestMetrics):
    _metrics: metrics.Metrics = metrics.GraphEntropyMetrics()


class TestGraphReconstructionEntropy(unittest.TestCase, TemplateTestMetrics):
    _metrics: metrics.Metrics = metrics.GraphReconstructionEntropyMetrics()
    not_implemented: bool = True


class TestReconstructability(unittest.TestCase, TemplateTestMetrics):
    _metrics: metrics.Metrics = metrics.ReconstructabilityMetrics()
    not_implemented: bool = True


class TestPredictability(unittest.TestCase, TemplateTestMetrics):
    _metrics: metrics.Metrics = metrics.PredictabilityMetrics()
    not_implemented: bool = True


if __name__ == "__main__":
    unittest.main()
