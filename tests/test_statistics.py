import pytest
import numpy as np
import pandas as pd

from midynet.statistics import Statistics


@pytest.fixture
def samples():
    return np.random.randn(100)


@pytest.fixture
def loc():
    return np.random.randn(3)


@pytest.fixture
def scale():
    return np.random.randn(3)


def test_statistics_constructor(loc, scale):
    stat = Statistics(dict(loc=loc, scale=scale))


@pytest.fixture
def stat(loc, scale):
    return Statistics(dict(loc=loc, scale=scale))


def test_statistics_shape(stat, loc, scale):
    assert stat.shape == loc.shape == scale.shape


def test_statistics_dict(stat, loc, scale):
    d = dict(stat_loc=loc, stat_scale=scale)
    assert stat.dict == d


def test_statistics_len(stat):
    assert len(stat) == len(stat.__data__)


@pytest.fixture
def other_stat():
    loc, scale = (np.random.randn(3),) * 2
    return Statistics(dict(loc=loc, scale=scale))


def test_statistics_concat(stat, other_stat):
    concat_stat = stat.concat(other_stat)
    assert concat_stat.shape == (stat.shape[0] + other_stat.shape[0],)


def test_statistics_bootstrap(stat):
    bs = stat.bootstrap(size=1000)
    assert bs.shape == (*stat.shape, 1000)
    assert np.allclose(bs.mean(-1), stat.loc, rtol=0.1)


if __name__ == "__main__":
    pass
