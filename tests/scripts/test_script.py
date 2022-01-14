import unittest

from midynet import config
from midynet import scripts


class TestScriptManager(unittest.TestCase):
    def setUp(self):
        self.config = config.ExperimentConfig.default("test", "sis", "uniform_sbm")
        self.script = scripts.ScriptManager(
            "test_script",
            "python run.py",
            config=self.config,
            env_to_load="./env/midynet-env/bin/activate",
        )

    def test_write_script(self):
        print(self.script.write_script(self.config))

    def test_set_up_script(self):
        pass

    def test_tear_down_script(self):
        pass

    def test_run(sel):
        pass

    def test_split_param(self):
        pass


if __name__ == "__main__":
    unittest.main()
