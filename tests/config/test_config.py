import midynet as md
import unittest

from midynet.config import *


class TestConfig(unittest.TestCase):
    display: bool = False

    def setUp(self):
        self.x: int = 1
        self.y: float = 0.5
        self.z: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.w: str = 0.5
        self.config = md.config.Config(x=self.x, y=self.y, z=self.z, w=self.w)
        self.r_config = md.config.Config(config=self.config, other=self.x)

    def test_keys(self):
        self.assertEqual(list(self.config.keys()), ["x", "y", "z", "w"])

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
        self.assertEqual(self.config.get("x"), self.r_config.get("config/x"))

    def test_dictcopy_recursively(self):
        self.assertEqual(len(self.r_config.dict_copy(recursively=True)), 6)
        for expected in [
            "config",
            "config/x",
            "config/y",
            "config/z",
            "config/w",
            "other",
        ]:
            self.assertIn(expected, self.r_config.dict_copy(recursively=True))

    def test_show(self):
        if self.display:
            print()
            print(self.r_config.format())

    def test_generate_sequence(self):
        counter = 0
        for c in self.config.generate_sequence():
            counter += 1
            self.assertFalse(c.has_sequence())
            if self.display:
                print()
                print(c.format())
        self.assertEqual(counter, len(self.z))

    def test_generate_sequence_recursive(self):
        counter = 0
        for c in self.r_config.generate_sequence():
            counter += 1
            self.assertFalse(c.has_sequence())
            if self.display:
                print()
                print(c.format())
        self.assertEqual(counter, len(self.z))

    def test_is_equivalent(self):
        self.assertTrue(self.config.is_equivalent(self.r_config["config"].value))

    def test_is_subconfig(self):
        config = Config(x=self.x, y=self.y, z=self.z[0], w=self.w)
        r_config = Config(config=config, other=self.x)
        self.assertTrue(self.config.is_subconfig(config))
        self.assertTrue(self.r_config.is_subconfig(r_config))

        config = Config(x=self.x, y=self.y, z=-1, w=self.w)
        r_config = Config(config=config, other=self.x)
        self.assertFalse(self.config.is_subconfig(config))
        self.assertFalse(self.r_config.is_subconfig(r_config))


if __name__ == "__main__":
    unittest.main()