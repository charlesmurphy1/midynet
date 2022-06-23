import pathlib

import pytest

from midynet.config import (
    Config,
    Parameter,
    MetricsConfig,
    MetricsCollectionConfig,
    ExperimentConfig,
)

DISPLAY = False


@pytest.fixture
def config():
    x = 1
    y = 0.5
    z = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    w = 0.5
    return Config(name="config", x=x, y=y, z=z, w=w)


@pytest.fixture
def r_config(config):
    return Config(name="r_config", config=config, other=config.x)


@pytest.fixture
def m_config():
    return Config(
        name="m_config",
        x=[
            Config(name="x_a", a=[1, 2, 3]),
            Config(name="x_b", b=2),
        ],
        y=[-1, 0, 1],
    )


def test_baseconfig_keys(config):
    for k in config.keys():
        assert k in ["name", "x", "y", "z", "w"]


def test_baseconfig_values(config):
    numberOfElements = 0
    for v in config.values():
        assert isinstance(v, Parameter)
        numberOfElements += 1
    assert numberOfElements == 5


def test_baseconfig_getitem(config):
    for k in ["x", "y", "z", "w"]:
        assert isinstance(config[k], Parameter)
    assert config["x"].value == 1
    assert config["y"].value == 0.5
    assert config["z"].value == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert config["w"].value == 0.5


def test_baseconfig_contains(config):
    for k in ["x", "y", "z", "w"]:
        assert k in config


def test_baseconfig_get_recursively(config, r_config):
    assert config.get_param("x") == r_config.get_param(f"config{Config.separator}x")


def test_baseconfig_dictcopy_recursively(r_config):
    assert len(r_config.dict_copy()) == 8
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
        assert expected in r_config.dict_copy()


def test_baseconfig_format(r_config):
    if DISPLAY:
        print()
        print(r_config.format())


def test_baseconfig_generate_sequence(config):
    counter = 0
    for c in config.sequence():
        counter += 1
        assert not c.is_sequenced()
        if DISPLAY:
            print()
            print(c.format())
    assert counter == len(config.z)


def test_baseconfig_generate_sequence_recursive(r_config):
    counter = 0
    for c in r_config.sequence():
        counter += 1
        assert not c.is_sequenced()
        if DISPLAY:
            print()
            print(c.format())
    assert counter == len(r_config.config.z)


def test_baseconfig_generate_sequence_with_muliple_subconfigs(m_config):
    counter = 0
    names = set()
    for cc in m_config.sequence():
        counter += 1
        names.add(cc.name)
        if DISPLAY:
            print()
            print(cc.format())

    assert counter == 12
    assert names == m_config.names()


def test_baseconfig_is_equivalent(config, r_config):
    assert config.is_equivalent(r_config["config"].value)
    c = config.copy()
    c.set_value("name", "c")


def test_baseconfig_is_subconfig(config, r_config):
    config = Config(x=config.x, y=config.y, z=1, w=config.w)
    assert config.is_subconfig(config)
    r_config = Config(config=config, other=config.x)
    assert r_config.is_subconfig(r_config)


def test_baseconfig_is_subset(m_config):
    for c in m_config.sequence():
        assert c.is_subset(m_config)


def test_baseconfig_scanned_keys(m_config):
    if DISPLAY:
        print(m_config.scanned_keys())


def test_baseconfig_scanned_values(m_config):
    if DISPLAY:
        print(m_config.scanned_values())


def test_baseconfig_hash_dict(m_config):
    if DISPLAY:
        print(m_config.hash_dict())
        for c in m_config.sequence():
            assert hash(c) in m_config.hash_dict()[c.name]


def test_baseconfig_merge_nonsequence_configs():
    c1 = Config(name="c1", x=1, y=4)
    c2 = Config(name="c2", x=2, y=4)

    c1.merge_with(c2)

    assert c1.name == "c1"
    assert c1.x == [1, 2]
    assert c1.y == 4

    assert c2.name == "c2"
    assert c2.x == 2
    assert c2.y == 4


def test_baseconfig_merge_sequence_configs():
    c1 = Config(name="c1", x=[1, 2], y=4)
    c2 = Config(name="c2", x=[3, 4], y=5)
    c1.merge_with(c2)

    assert c1.name == "c1"
    assert c1.x == [1, 2, 3, 4]
    assert c1.y == [4, 5]

    assert c2.name == "c2"
    assert c2.x == [3, 4]
    assert c2.y == 5


def test_baseconfig_merge_sequence_multiconfigs():

    c1 = Config(name="c1", x=[1, 2], y=4)
    c2 = Config(name="c2", x=[3, 4], y=5)
    c3 = Config(name="c3", c=c1, z=4)
    c4 = Config(name="c4", c=c2, z=5)
    c5 = c4.copy()
    c5.merge_with(c3)
    assert c3.is_subset(c5)
    assert c4.is_subset(c5)
    assert not c2.is_subset(c5)


def test_baseconfig_name_intricate_multiconfig():
    bottom1 = Config(name="bottom1", x=[1, 2], y=4)
    bottom2 = Config(name="bottom2", x=[3, 4], y=5)
    middle = Config(name="middle", bottom=[bottom1, bottom2], z=2)
    midtop = Config(name="midtop", middle=middle)

    other1 = Config(name="other1", x=[1, 2], y=4)
    other2 = Config(name="other2", x=[3, 4], y=5)
    top = Config(name="top", midtop=midtop, other=[other1, other2])
    if DISPLAY:
        print(top.format())
        for c in top.sequence():
            print(c.name)


def test_baseconfig_save(config):
    path = pathlib.Path("./test_config.pickle")
    config.save(path)
    assert path.exists()
    path.unlink()
    assert not path.exists()


def test_baseconfig_load(m_config):
    path = pathlib.Path("./test_config.pickle")
    m_config.save(path)
    c = Config.load(path)
    assert m_config.is_equivalent(c)
    path.unlink()


def test_metrics_config_auto():
    metrics = MetricsConfig.auto("dynamics_entropy")
    assert len(metrics.sequence()) == 1


def test_metrics_collection_config_auto():
    metrics_config = MetricsCollectionConfig.auto("dynamics_entropy")
    assert "dynamics_entropy" in metrics_config
    assert "dynamics_entropy" in metrics_config.metrics_names

    metrics_config = MetricsCollectionConfig.auto(["dynamics_entropy", "graph_entropy"])
    assert "dynamics_entropy" in metrics_config
    assert "graph_entropy" in metrics_config
    assert "dynamics_entropy" in metrics_config.metrics_names
    assert "graph_entropy" in metrics_config.metrics_names
    assert not metrics_config.is_sequenced()
    assert len(metrics_config.sequence()) == 1


def test_experiment_config_reconstruction():
    exp = ExperimentConfig.reconstruction(
        "test",
        "sis",
        "uniform_sbm",
        metrics=["dynamics_entropy", "graph_entropy"],
    )
    assert "name" in exp
    assert "dynamics" in exp
    assert "graph" in exp
    assert "metrics" in exp
    assert "path" in exp
    assert "seed" in exp
    assert "num_procs" in exp
    assert len(exp) == 1
    assert len(exp.unmet_requirements()) == 0


if __name__ == "__main__":
    pass
