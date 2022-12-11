import pathlib

import pytest

from midynet.config import (
    Config,
    GraphConfig,
    DataModelConfig,
    ExperimentConfig,
)


@pytest.fixture
def config():
    return Config(name="config", x=1, y=2)


def test_baseconfig_for_attributes(config):
    assert "x" in config and config.x == 1
    assert "y" in config and config.y == 2


def test_baseconfig_is_sequenced(config):
    assert not config.is_sequenced()


def test_baseconfig_to_sequenc(config):
    seq = list(config.to_sequence())
    assert len(seq) == 1
    assert seq[0].name == "config"


def test_baseconfig_types(config):
    config.x = 10
    assert config.__types__["x"] == int

    config.x = [1, 2, 3, 4, 5, 6, 7]
    assert config.__types__["x"] == int

    config.x = 1.1
    assert config.__types__["x"] == float


def test_baseconfig_in_lock_types(config):
    config.x = 10
    assert config.__types__["x"] == int

    config.lock_types()

    config.x = [1, 2, 3, 4, 5, 6, 7]
    assert config.__types__["x"] == int

    with pytest.raises(Exception):
        config.x = 1.1

    with pytest.raises(Exception):
        config.new_params = "not ok"


def test_baseconfig_set_attributes_with_multiple_types_raises_valueerror(
    config,
):
    with pytest.raises(ValueError):
        config.x = [1, "k"]


@pytest.fixture
def nested_config():
    c = Config("sub", x=1, y=2)
    return Config(name="base", z=3, c=c)


def test_get_with_nestedconfig(nested_config):
    assert nested_config.get("z") == 3
    assert nested_config.get("x") is None
    assert nested_config.get("c.x") == 1


def test_nestedconfig_for_attributes(nested_config):
    assert "z" in nested_config and nested_config.z == 3
    assert "c" in nested_config
    assert "x" in nested_config.c and nested_config.c.x == 1
    assert "y" in nested_config.c and nested_config.c.y == 2


def test_nestedconfig_is_sequenced(nested_config):
    assert not nested_config.is_sequenced()


def test_nestedconfig_to_sequence(nested_config):
    seq = list(nested_config.to_sequence())
    assert len(seq) == 1
    assert seq[0].name == "base"


@pytest.fixture
def sequenced_config():
    return Config(name="sequenced_config", x=1, y=[0, 1, 2, 3])


def test_sequencedconfig_for_attributes(sequenced_config):
    assert "x" in sequenced_config and sequenced_config.x == 1
    assert "y" in sequenced_config and len(sequenced_config.y) == 4
    for exp, act in zip([0, 1, 2, 3], sequenced_config.y):
        assert exp == act


def test_sequencedconfig_is_sequenced(sequenced_config):
    assert sequenced_config.is_sequenced()


def test_sequencedconfig_to_sequence(sequenced_config):
    seq = list(sequenced_config.to_sequence())
    assert len(seq) == 4
    for s in seq:
        assert s.name == "sequenced_config"


@pytest.fixture
def sequenced_nested_config():
    c1 = Config(name="case1", x=1, y=[0, 1, 2, 3])
    c2 = Config(name="case2", xx=1, yy=2)
    return Config(name="base", z=3, c=[c1, c2])


def test_sequencednestedconfig_for_attributes(sequenced_nested_config):
    assert "z" in sequenced_nested_config and sequenced_nested_config.z == 3
    assert (
        "c" in sequenced_nested_config and len(sequenced_nested_config.c) == 2
    )
    for c in sequenced_nested_config.c:
        assert issubclass(c.__class__, Config)


def test_sequencednestedconfig_is_sequenced(sequenced_nested_config):
    assert sequenced_nested_config.is_sequenced()


def test_sequencednestedconfig_to_sequence(sequenced_nested_config):
    seq = list(sequenced_nested_config.to_sequence())
    assert len(seq) == 5
    for s in seq:
        assert len(s.name.split(Config.separator)) == 2


def test_sequencednestedconfig_dict_and_back(sequenced_nested_config):
    c = Config.from_dict(sequenced_nested_config.dict)
    assert c.dict == sequenced_nested_config.dict


def test_highly_nested_config():
    c1 = Config(name="case1", x=1, y=[0, 1, 2, 3])
    c2 = Config(name="case2", x=1, y=[0, 1, 2, 3])
    c3 = Config(name="case3", xx=1, yy=2, c=c2)
    config = Config(name="base", z=3, c=[c1, c3], k=[1, 2, 3, 4])
    for c in config.to_sequence():
        assert isinstance(c, Config)


@pytest.mark.parametrize(
    "config",
    [
        GraphConfig.erdosrenyi,
        # GraphConfig.configuration,
        # GraphConfig.stochastic_block_model,
    ],
)
def test_graph_config(config):
    c = config()
    assert "name" in c
    assert "size" in c
    assert "edge_proposer_type" in c
    assert c.name == config.__name__

    # assert that config is static
    with pytest.raises(Exception):
        c.new_params = "hello world"
    with pytest.raises(Exception):
        c.size = "hello world"
    c.size = 12
    c.size = [12, 24, 64]


@pytest.mark.parametrize(
    "config",
    [
        DataModelConfig.glauber,
        DataModelConfig.sis,
        DataModelConfig.cowan,
    ],
)
def test_datamodel_config(config):
    c = config()
    assert "name" in c
    assert c.name == config.__name__

    # assert that config is static
    with pytest.raises(Exception):
        c.new_params = "hello world"
    with pytest.raises(Exception):
        c.length = "hello world"
    c.length = 12
    c.length = [12, 24, 64]


# def test_exp_config():
#     c = ExperimentConfig.reconstruction("test", "sis", "erdosrenyi")
#     assert "data_model" in c and issubclass(c.data_model.__class__, Config)
#     assert "prior" in c and issubclass(c.prior.__class__, Config)

#     c = ExperimentConfig.reconstruction(
#         "test",
#         ["sis", "glauber", "cowan"],
#         ["erdosrenyi", "configuration", "stochastic_block_model"],
#     )
#     assert "data_model" in c and issubclass(
#         c.data_model.__class__, ParameterSequence
#     )
#     assert "prior" in c and issubclass(c.prior.__class__, ParameterSequence)
#     print(c)


if __name__ == "__main__":
    pass
