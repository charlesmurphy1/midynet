import midynet as md
import typing
import pathlib
import unittest

from midynet.config import *


class TestConfig(unittest.TestCase):
    display: bool = False

    def setUp(self):
        self.x: int = 1
        self.y: float = 0.5
        self.z: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.w: str = 0.5
        self.config = md.config.Config(
            name="config", x=self.x, y=self.y, z=self.z, w=self.w
        )
        self.r_config = md.config.Config(
            name="r_config", config=self.config, other=self.x
        )

        self.m_config = config.Config(
            name="m_config",
            x=[
                config.Config(name="x_a", a=[1, 2, 3]),
                config.Config(name="x_b", b=2),
            ],
            y=[-1, 0, 1],
        )

    def test_keys(self):
        for k in self.config.keys():
            self.assertIn(k, ["name", "x", "y", "z", "w"])

    def test_values(self):
        numberOfElements = 0
        for v in self.config.values():
            self.assertTrue(isinstance(v, Parameter))
            numberOfElements += 1
        self.assertEqual(numberOfElements, 5)

    def test_getitem(self):
        for k in ["x", "y", "z", "w"]:
            self.assertTrue(isinstance(self.config[k], Parameter))
            self.assertEqual(self.config[k].value, getattr(self, k))

    def test_contains(self):
        for k in ["x", "y", "z", "w"]:
            self.assertIn(k, self.config)

    def test_get_recursively(self):
        self.assertEqual(
            self.config.get_param("x"),
            self.r_config.get_param(f"config{Config.separator}x"),
        )

    def test_dictcopy_recursively(self):
        self.assertEqual(len(self.r_config.dict_copy()), 8)
        for expected in [
            "name",
            "config",
            f"config{Config.separator}name",
            f"config{Config.separator}x",
            f"config{Config.separator}y",
            f"config{Config.separator}z",
            f"config{Config.separator}w",
            "other",
        ]:
            self.assertIn(expected, self.r_config.dict_copy())

    def test_format(self):
        if self.display:
            print()
            print(self.r_config.format())

    def test_generate_sequence(self):
        counter = 0
        for c in self.config.generate_sequence():
            counter += 1
            self.assertFalse(c.is_sequenced())
            if self.display:
                print()
                print(c.format())
        self.assertEqual(counter, len(self.z))

    def test_generate_sequence_recursive(self):
        counter = 0
        for c in self.r_config.generate_sequence():
            counter += 1
            self.assertFalse(c.is_sequenced())
            if self.display:
                print()
                print(c.format())
        self.assertEqual(counter, len(self.z))

    def test_generate_sequence_with_muliple_subconfigs(self):
        counter = 0
        names = set()
        for cc in self.m_config.generate_sequence():
            counter += 1
            names.add(cc.name)
            if self.display:
                print()
                print(cc.format())

        self.assertEqual(counter, 12)
        self.assertEqual(names, self.m_config.names)

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

    def test_scanned_keys(self):
        if self.display:
            print(self.m_config.scanned_keys)

    def test_scanned_values(self):
        if self.display:
            print(self.m_config.scanned_values)

    def test_merge_nonsequence_configs(self):
        c1 = Config(name="c1", x=1, y=4)
        c2 = Config(name="c2", x=2, y=4)
        c1.merge(c2)

        self.assertEqual(c1.name, "c1")
        self.assertEqual(c1.x, {1, 2})
        self.assertEqual(c1.y, 4)

        self.assertEqual(c2.name, "c2")
        self.assertEqual(c2.x, 2)
        self.assertEqual(c2.y, 4)

    def test_merge_sequence_configs(self):
        c1 = Config(name="c1", x={1, 2}, y=4)
        c2 = Config(name="c2", x={3, 4}, y=5)
        c1.merge(c2)

        self.assertEqual(c1.name, "c1")
        self.assertEqual(c1.x, {1, 2, 3, 4})
        self.assertEqual(c1.y, {4, 5})

        self.assertEqual(c2.name, "c2")
        self.assertEqual(c2.x, {3, 4})
        self.assertEqual(c2.y, 5)

    def test_save(self):
        path = pathlib.Path("./test_config.pickle")
        self.config.save(path)
        self.assertTrue(path.exists())
        path.unlink()
        self.assertFalse(path.exists())

    def test_load(self):
        path = pathlib.Path("./test_config.pickle")
        self.m_config.save(path)
        c = Config.load(path)
        self.assertTrue(self.m_config.is_equivalent(c))
        path.unlink()


if __name__ == "__main__":
    unittest.main()
