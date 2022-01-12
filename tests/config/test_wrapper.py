import unittest

from midynet import config
from _midynet.prior import sbm


class TestWrapper(unittest.TestCase):
    def setUp(self):
        self.config = config.BlockPriorConfig.uniform(100)
        block_count = sbm.BlockCountUniformPrior(10)
        blocks = sbm.BlockUniformPrior(self.config.size, block_count)
        self.wrapper = config.Wrapper(
            blocks,
            setup_func=lambda wrap, other: wrap.set_block_count_prior(
                other["block_count"]
            ),
            block_count=block_count,
        )

    def test_access_wrapped_method(self):
        self.assertEqual(self.wrapper.get_size(), 100)
        self.wrapper.sample_priors()

    def test_get_wrap(self):
        isinstance(self.wrapper.get_wrap(), sbm.BlockUniformPrior)

    def test_get_others(self):
        isinstance(self.wrapper.get_others(), sbm.BlockCountUniformPrior)

    def test_correct_setup(self):
        self.assertEqual(
            id(self.wrapper.get_block_count_prior()),
            id(self.wrapper.get_others()["block_count"]),
        )


if __name__ == "__main__":
    unittest.main()
