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

    def test_build_good_config(self):
        for c in self.good_configs:
            self.factory.build(c)

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


class TestBlockCountPriorFactory(unittest.TestCase, TestFactory):
    factory = BlockCountPriorFactory
    good_configs = [
        BlockCountPriorConfig.delta(5),
        BlockCountPriorConfig.poisson(5),
    ]
    missing_configs = [Config(name="missing")]
    unavailable_configs = [BlockCountPriorConfig.uniform(5)]


if __name__ == "__main__":
    unittest.main()
