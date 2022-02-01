import pytest
from _midynet.prior import sbm

from midynet.config import *


@pytest.fixture
def wrapper():
    size = 100
    max_block_count = 10
    config = BlockPriorConfig.uniform()
    block_count = sbm.BlockCountUniformPrior(max_block_count)
    blocks = sbm.BlockUniformPrior(size, block_count)
    return Wrapper(
        blocks,
        block_count=block_count,
    )


def test_access_wrapped_method(wrapper):
    assert wrapper.get_size() == 100
    wrapper.sample_priors()


def test_get_wrap(wrapper):
    assert isinstance(wrapper.get_wrap(), sbm.BlockUniformPrior)


def test_get_others(wrapper):
    assert isinstance(wrapper.get_other("block_count"), sbm.BlockCountUniformPrior)


def test_correct_setup(wrapper):
    assert id(wrapper.get_block_count_prior()) == id(
        wrapper.get_others()["block_count"]
    )


if __name__ == "__main__":
    pass
