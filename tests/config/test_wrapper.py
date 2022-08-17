import pytest

from _midynet.random_graph.prior import BlockCountUniformPrior, BlockUniformPrior
from midynet.wrapper import Wrapper


@pytest.fixture
def wrapper():
    size = 100
    max_block_count = 10
    block_count = BlockCountUniformPrior(1, max_block_count)
    blocks = BlockUniformPrior(size, block_count)
    return Wrapper(
        blocks,
        block_count=block_count,
    )


def test_access_wrapped_method(wrapper):
    assert wrapper.get_size() == 100
    wrapper.sample_priors()


def test_wrap(wrapper):
    assert isinstance(wrapper.wrap, BlockUniformPrior)


def test_other(wrapper):
    assert isinstance(wrapper.others["block_count"], BlockCountUniformPrior)
    assert isinstance(wrapper.other("block_count"), BlockCountUniformPrior)
    assert isinstance(wrapper.block_count, BlockCountUniformPrior)


def test_correct_setup(wrapper):
    assert id(wrapper.get_block_count_prior()) == id(wrapper.block_count)


if __name__ == "__main__":
    pass
