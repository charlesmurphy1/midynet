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

        # name: str
        # executable: str
        # config: Config = field(repr=False, default_factory=Config)
        # config_array: list[Config] = field(repr=False, default_factory=list)
        # execution_command: str = field(repr=False, default="bash")
        # resources: dict[str, str] = field(repr=False, default_factory=dict)
        # resource_prefix: str = field(repr=False, default="#SBATCH")
        # path_to_script: pathlib.Path = field(repr=False, default_factory=pathlib.Path)
        # path_to_scripts: list[pathlib.Path] = field(repr=False, default_factory=list)
        # modules_to_load: list[str] = field(repr=False, default_factory=list)
        # env_to_load: str = None

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
