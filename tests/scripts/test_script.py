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
            config=self.config,
            resources=resources,
            path_to_scripts=f"./tests/scripts/test-dir",
            env_to_load="./env/midynet-env/bin/activate",
            modules_to_load=["g++", "mpi", "other_modules"],
        )

    def tearDown(self):
        self.script.path_to_scripts.rmdir()

    def test_write_script(self):
        script = self.script.write_script(self.config, "test-007")
        if self.display:
            print(script)

    def test_set_up_script(self):
        tag = self.script.set_up(self.config)
        path_to_config = self.script.path_to_scripts / f"{tag}-config.pickle"
        self.assertTrue(path_to_config.exists())
        self.assertTrue(path_to_config.is_file())

        path_to_script = self.script.path_to_scripts / f"{tag}.sh"
        self.assertTrue(path_to_script.exists())
        self.assertTrue(path_to_script.is_file())
        path_to_config.unlink()
        path_to_script.unlink()

    def test_tear_down_script(self):
        tag = self.script.set_up(self.config)
        path_to_config = self.script.path_to_scripts / f"{tag}-config.pickle"
        path_to_script = self.script.path_to_scripts / f"{tag}.sh"

        self.script.tear_down(tag)
        self.assertFalse(path_to_config.exists())
        self.assertFalse(path_to_script.exists())
        self.assertTrue(self.script.path_to_scripts.exists())

    def test_run(sel):
        pass

    def test_split_param(self):
        if self.display:
            print(self.script.config.format())

        self.script.split_param("dynamics")

        if self.display or True:
            for c in self.script.config_array:
                print(c.format())


if __name__ == "__main__":
    unittest.main()
