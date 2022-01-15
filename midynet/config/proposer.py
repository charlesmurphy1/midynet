from __future__ import annotations

from .config import Config
from .factory import Factory
from .wrapper import Wrapper
from _midynet import proposer

__all__ = [
    "EdgeProposerConfig",
    "BlockProposerConfig",
    "EdgeProposerFactory",
    "BlockProposerFactory",
]


class EdgeProposerConfig(Config):
    @classmethod
    def single_uniform(cls):
        return cls(name="single_uniform")

    @classmethod
    def single_degree(cls):
        return cls(name="single_degree", shift=1)

    @classmethod
    def hinge_flip_uniform(cls):
        return cls(name="hinge_flip_uniform")

    @classmethod
    def hinge_flip_degree(cls):
        return cls(name="hinge_flip_degree", shift=1)

    @classmethod
    def double_swap(cls):
        return cls(name="double_swap")


class BlockProposerConfig(Config):
    @classmethod
    def uniform(cls):
        return cls(name="uniform", create_new_block=0.1)

    @classmethod
    def peixoto(cls):
        return cls(name="peixoto", create_new_block=0.1, shift=1)


class EdgeProposerFactory(Factory):
    @staticmethod
    def build_single_uniform(
        config: EdgeProposerConfig,
    ) -> proposer.edge_proposer.SingleEdgeUniformProposer:
        return proposer.edge_proposer.SingleEdgeUniformProposer()

    @staticmethod
    def build_single_degree(
        config: EdgeProposerConfig,
    ) -> proposer.edge_proposer.SingleEdgeDegreeProposer:
        return proposer.edge_proposer.SingleEdgeDegreeProposer(config.shift)

    @staticmethod
    def build_hinge_flip_uniform(
        config: EdgeProposerConfig,
    ) -> proposer.edge_proposer.HingeFlipUniformProposer:
        return proposer.edge_proposer.HingeFlipUniformProposer()

    @staticmethod
    def build_hinge_flip_degree(
        config: EdgeProposerConfig,
    ) -> proposer.edge_proposer.HingeFlipDegreeProposer:
        return proposer.edge_proposer.HingeFlipDegreeProposer(config.shift)

    @staticmethod
    def build_double_swap(
        config: EdgeProposerConfig,
    ) -> proposer.edge_proposer.DoubleEdgeSwapProposer:
        return proposer.edge_proposer.DoubleEdgeSwapProposer()


class BlockProposerFactory(Factory):
    @staticmethod
    def build_uniform(
        config: BlockProposerFactory,
    ) -> proposer.block_proposer.UniformBlockProposer:
        return proposer.block_proposer.UniformBlockProposer(config.create_new_block)

    @staticmethod
    def build_peixoto(
        config: BlockProposerFactory,
    ) -> proposer.block_proposer.PeixotoBlockProposer:
        return proposer.block_proposer.PeixotoBlockProposer(
            config.create_new_block, config.shift
        )
