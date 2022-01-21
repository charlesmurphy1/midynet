import unittest
import pathlib
import os

from midynet import config
from midynet import scripts


class TestScriptManager(unittest.TestCase):
    display: bool = False

    def setUp(self):
        self.config = config.ExperimentConfig.default(
            "test", ["ising", "sis"], "uniform_sbm"
        )
        resources = {
            "account": "def-aallard",
            "time": "24:00:00",
            "output": "./test",
            "mem": "12G",
            "cpus-per-task": "40",
        }
        self.script = scripts.ScriptManager(
            "test_script",
            "python run.py",
            path_to_scripts=f"./tests/scripts/test-dir",
        )

    def tearDown(self):
        self.script.path_to_scripts.rmdir()

    def test_write_script(self):
        script = self.script.write_script(self.config, nametag="test-007", resources={})
        if self.display:
            print(script)

    def test_set_up_script(self):
        nametag = self.script.set_up(self.config, tag=1)
        path_to_config = self.script.path_to_scripts / f"{nametag}-config.pickle"
        self.assertTrue(path_to_config.exists())
        self.assertTrue(path_to_config.is_file())

        path_to_script = self.script.path_to_scripts / f"{nametag}.sh"
        self.assertTrue(path_to_script.exists())
        self.assertTrue(path_to_script.is_file())
        path_to_config.unlink()
        path_to_script.unlink()

    def test_tear_down_script(self):
        nametag = self.script.set_up(self.config, tag=1)
        path_to_config = self.script.path_to_scripts / f"{nametag}-config.pickle"
        path_to_script = self.script.path_to_scripts / f"{nametag}.sh"
        self.script.tear_down(nametag)
        self.assertFalse(path_to_config.exists())
        self.assertFalse(path_to_script.exists())
        self.assertTrue(self.script.path_to_scripts.exists())

    def test_run(sel):
        pass

    def test_split_param(self):
        if self.display:
            print(self.script.config.format())

        configs = self.script.split_param(self.config, "dynamics")

        if self.display:
            for c in configs:
                print(c.format())


if __name__ == "__main__":
    unittest.main()
