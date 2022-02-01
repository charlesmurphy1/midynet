import pytest

from midynet import config


def test_datatype():
    p = config.Parameter(name="x", value=10)
    assert p.datatype == int

    p = config.Parameter(name="x", value=10.0)
    assert p.datatype == float

    p = config.Parameter(name="x", value=[1, 2, 3, 4, 5])
    assert p.datatype == int


def test_getitem():
    p = config.Parameter(name="test", value=1)
    with pytest.raises(LookupError):
        p[0]

    p = config.Parameter(name="test", value=[1, 2, 3, 4, 5])
    assert p[0] == 1


def test_set_value():
    p = config.Parameter(name="test", value=1)
    p.set_value(2.5)
    assert p.datatype == float
    assert p.value == 2.5


def test_is_sequenced():
    p = config.Parameter(name="test", value=1)
    assert not p.is_sequenced()

    p = config.Parameter(name="test", value=[1, 2, 3, 4, 5])
    assert p.is_sequenced()


def test_is_unique():
    p = config.Parameter(name="test", value=1, unique=False)
    assert not p.is_unique()

    p = config.Parameter(name="test", value=1, unique=True)
    assert p.is_unique()


def test_infer_type():
    p = config.Parameter(name="test", value=1)

    assert p.infer_type(1) == int
    assert p.infer_type([1]) == int
    assert p.infer_type([[1]]) == int
    assert p.infer_type([(1,)]) == int

    p = config.Parameter(name="test", value=1, force_non_sequence=True)
    assert p.infer_type([1]) == list


def test_generate_sequence():
    p = config.Parameter(name="test", value=[1, 2, 3, 4, 5])
    for pp, x in zip(p.generate_sequence(), [1, 2, 3, 4, 5]):
        assert pp == x


def test_add_value_to_nonsequenced_parameter():
    p = config.Parameter(name="x", value=1)
    assert not p.is_sequenced()

    p.add_value(2)
    assert p.value == [1, 2]
    assert p.is_sequenced()


def test_add_value_to_sequenced_parameter():
    p = config.Parameter(name="x", value=[1.1, 2])
    assert p.is_sequenced()

    p.add_value(2)
    assert p.value == [1.1, 2]

    p.add_value(3)
    assert p.value == [1.1, 2, 3]


def test_add_values_to_nonsequenced_parameter():
    p = config.Parameter(name="x", value=1.1)
    assert not p.is_sequenced()

    p.add_values([2, 3])
    assert p.value == [1.1, 2, 3]
    assert p.is_sequenced()


def test_add_values_to_sequenced_parameter():
    p = config.Parameter(name="x", value=[1.1, 2])
    assert p.is_sequenced()

    p.add_values([2, 3])
    assert p.value == [1.1, 2, 3]


if __name__ == "__main__":
    pass
