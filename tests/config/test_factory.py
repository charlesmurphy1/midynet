import midynet as md
import unittest

from dataclasses import dataclass, field
from typing import Type

from midynet.config import *


class TestFactory:
    factory: Type[Factory]
    good_configs: list[Config] = []
    missing_configs: list[Config] = []
    unavailable_configs: list[Config] = []
    run_sample: bool = False

    def obj_testing(self, obj):
        pass

    def test_build_good_config(self):
        for c in self.good_configs:
            obj = self.factory.build(c)
            self.obj_testing(obj)

    def test_build_missing_config(self):
        for c in self.missing_configs:
            with self.assertRaises(OptionError):
                self.factory.build(c)

    def test_build_unavailable_config(self):
        for c in self.unavailable_configs:
            with self.assertRaises(NotImplementedError):
                self.factory.build(c)


class TestEdgeCountPriorFactory(unittest.TestCase, TestFactory):
    factory = EdgeCountPriorFactory
    good_configs = [
        EdgeCountPriorConfig.delta(5),
        EdgeCountPriorConfig.poisson(5),
    ]
    missing_configs = [Config(name="missing")]
    run_sample: bool = False

    def obj_testing(self, obj):
        obj.sample()


class TestBlockCountPriorFactory(unittest.TestCase, TestFactory):
    factory = BlockCountPriorFactory
    good_configs = [
        BlockCountPriorConfig.delta(5),
        BlockCountPriorConfig.poisson(5),
        BlockCountPriorConfig.uniform(5),
    ]
    missing_configs = [Config(name="missing")]

    def obj_testing(self, obj):
        obj.sample()


class TestBlockPriorFactory(unittest.TestCase, TestFactory):
    factory = BlockPriorFactory
    good_configs = [
        BlockPriorConfig.delta([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        BlockPriorConfig.uniform(),
        BlockPriorConfig.hyperuniform(),
    ]
    missing_configs = [Config(name="missing")]

    def obj_testing(self, obj):
        obj.sample()


class TestEdgeMatrixPriorFactory(unittest.TestCase, TestFactory):
    factory = EdgeMatrixPriorFactory
    good_configs = [
        EdgeMatrixPriorConfig.uniform(10),
    ]

    def obj_testing(self, obj):
        c = BlockPriorConfig.uniform()
        c.block_count.max = 10
        self.b = BlockPriorFactory.build(BlockPriorConfig.uniform())
        obj.set_block_prior(self.b.get_wrap())
        obj.sample()


class TestDegreePriorFactory(unittest.TestCase, TestFactory):
    factory = DegreePriorFactory
    good_configs = [
        DegreePriorConfig.uniform(),
    ]
    unavailable_configs = [DegreePriorConfig.hyperuniform()]

    def setUp_object(self, obj):
        self.b = BlockPriorFactory.build(BlockPriorConfig.uniform(100))

        self.e = EdgeMatrixPriorFactory.build(EdgeMatrixPriorConfig.uniform(250))
        self.e.set_block_prior(self.b.get_wrap())

        obj.set_block_prior(self.b.get_wrap())
        obj.set_edge_matrix_prior(self.e.get_wrap())
        obj.sample()


class TestRandomGraphFactory(unittest.TestCase, TestFactory):
    factory = RandomGraphFactory
    good_configs = {
        RandomGraphConfig.uniform_sbm(100, 250, 10),
        RandomGraphConfig.hyperuniform_sbm(100, 250, 10),
        RandomGraphConfig.fixed_er(100, 250),
        RandomGraphConfig.poisson_er(100, 250.0),
        RandomGraphConfig.uniform_dcsbm(100, 250, 10),
        RandomGraphConfig.uniform_cm(100, 250),
    }
    missing_configs = [Config(name="missing")]
    unavailable_configs = [
        RandomGraphConfig.hyperuniform_dcsbm(100, 250, 10),
        RandomGraphConfig.hyperuniform_cm(100, 250),
    ]

    def obj_testing(self, obj):
        obj.sample()


class TestDynamicsFactory(unittest.TestCase, TestFactory):
    factory = DynamicsFactory
    good_configs = {
        DynamicsConfig.ising(),
        DynamicsConfig.sis(),
        DynamicsConfig.cowan(),
        DynamicsConfig.degree(),
    }

    def obj_testing(self, obj):
        c = RandomGraphConfig.fixed_er(10, 25)
        g = RandomGraphFactory.build(c)
        obj.set_random_graph(g.get_wrap())
        obj.set_num_steps(10)
        obj.sample()


class TestRandomGraphMCMCFactory(unittest.TestCase, TestFactory):

    factory = RandomGraphMCMCFactory
    good_configs = {
        RandomGraphConfig.uniform_sbm(100, 250, 10),
        RandomGraphConfig.poisson_er(100, 250),
        RandomGraphConfig.fixed_er(100, 250),
        RandomGraphConfig.uniform_dcsbm(100, 250, 10),
        RandomGraphConfig.uniform_cm(100, 250),
    }


class TestMetricsFactory(unittest.TestCase, TestFactory):
    factory = MetricsFactory
    good_configs = {
        MetricsConfig.dynamics_entropy(),
        MetricsCollectionConfig.auto(["graph_entropy", "graph_entropy"]),
    }


if __name__ == "__main__":
    unittest.main()
