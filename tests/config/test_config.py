import midynet as md
import unittest

from midynet.config import *

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.x :int = 1
        self.y : float= 0.5
        self.z : list[int] = 0.5
        self.w : str = 0.5
        self.config = md.config.Config(x=self.x, y=self.y, z=self.z, w=self.w)
        self.recursive_config = md.config.Config(config=self.config, other=self.x)

    def test_keys(self):
        self.assertEqual(list(self.config.keys()), ["x", "y", "z", "w"])
    #
    def test_values(self):
        numberOfElements = 0
        for v in self.config.values():
            self.assertTrue(isinstance(v, Parameter))
            numberOfElements += 1
        self.assertEqual(numberOfElements, 4)

    def test_getitem(self):
        for k in self.config.keys():
            self.assertTrue(isinstance(self.config[k], Parameter))
            self.assertEqual(self.config[k].value, getattr(self, k))

    def test_contains(self):
        for k in ["x", "y", "z", "w"]:
            self.assertIn(k, self.config)

    def test_get_recursively(self):
        self.assertEqual(self.config.get("x"), self.recursive_config.get("config/x"))
