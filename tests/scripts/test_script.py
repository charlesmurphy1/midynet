import pytest
from midynet import scripts
from midynet.config import ExperimentConfig

DISPLAY = False


@pytest.fixture
def config():
    yield ExperimentConfig.reconstruction("test", ["glauber", "sis"], "erdosrenyi")


@pytest.fixture
def script():
    s = scripts.ScriptManager(
        "test_script",
        "python run.py",
        path_to_scripts="./testing/scripts/test-dir",
    )
    yield s
    path = s.path_to_scripts  # testing/scripts/test-dir
    path.rmdir()
    path = path.parent  # testing/scripts/
    path.rmdir()
    path = path.parent  # testing/
    path.rmdir()


def test_write_script(config, script):
    script = script.write_script(config, nametag="test-007", resources={})
    if DISPLAY:
        print(script)


def test_set_up_script(config, script):
    nametag = script.set_up(config, tag=1)
    path_to_config = script.path_to_scripts / f"{nametag}-config.pickle"
    assert path_to_config.exists()
    assert path_to_config.is_file()

    path_to_script = script.path_to_scripts / f"{nametag}.sh"
    assert path_to_script.exists()
    assert path_to_script.is_file()
    path_to_config.unlink()
    path_to_script.unlink()


def test_tear_down_script(config, script):
    nametag = script.set_up(config, tag=1)
    path_to_config = script.path_to_scripts / f"{nametag}-config.pickle"
    path_to_script = script.path_to_scripts / f"{nametag}.sh"
    script.tear_down(nametag)
    assert not path_to_config.exists()
    assert not path_to_script.exists()
    assert script.path_to_scripts.exists()


def test_run():
    pass


def test_split_param(config, script):
    if DISPLAY:
        print(script.config.format())

    configs = script.split_param(config, "data_model")

    if DISPLAY:
        for c in configs:
            print(c.format())


if __name__ == "__main__":
    pass
