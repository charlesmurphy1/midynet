import unittest

import midynet
from midynet.config import *


class TestDynamicsEntropy(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            name="test",
            graph=RandomGraphConfig.auto("poisson_er"),
            dynamics=DynamicsConfig.auto("sis"),
            metrics=Config(name="metrics", num_procs=4, num_samples=10),
        )
        self.metrics = DynamicsEntropyMetrics()

    def test_eval(self):
        print(self.metrics.eval(self.config))
