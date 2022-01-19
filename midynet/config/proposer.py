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
        return cls(name="single_uniform", allow_self_loops=True, allow_multiedges=True)

    @classmethod
    def single_degree(cls):
        return cls(
            name="single_degree", allow_self_loops=True, allow_multiedges=True, shift=1
        )

    @classmethod
    def hinge_flip_uniform(cls):
        return cls(
            name="hinge_flip_uniform", allow_self_loops=True, allow_multiedges=True
        )

    @classmethod
    def hinge_flip_degree(cls):
        return cls(
            name="hinge_flip_degree",
            allow_self_loops=True,
            allow_multiedges=True,
            shift=1,
        )

    @classmethod
    def double_swap(cls):
        return cls(name="double_swap", allow_self_loops=True, allow_multiedges=True)


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
        config: BlockProposerConfig,
    ) -> proposer.block_proposer.BlockUniformProposer:
        return proposer.block_proposer.BlockUniformProposer(config.create_new_block)

    @staticmethod
    def build_peixoto(
        config: BlockProposerConfig,
    ) -> proposer.block_proposer.BlockPeixotoProposer:
        return proposer.block_proposer.BlockPeixotoProposer(
            config.create_new_block, config.shift
        )

    @staticmethod
    def build_generic(
        config: BlockProposerConfig,
    ) -> proposer.block_proposer.BlockGenericProposer:
        return proposer.block_proposer.BlockGenericProposer()
