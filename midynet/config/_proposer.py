from __future__ import annotations

from _midynet import proposer

from .config import Config
from .factory import Factory

__all__ = (
    "EdgeProposerConfig",
    "BlockProposerConfig",
    "EdgeProposerFactory",
    "BlockProposerFactory",
)


class EdgeProposerConfig(Config):
    @classmethod
    def single_uniform(cls):
        return cls(name="single_uniform", allow_self_loops=True, allow_multiedges=True)

    @classmethod
    def single_degree(cls):
        return cls(
            name="single_degree",
            allow_self_loops=True,
            allow_multiedges=True,
            shift=1,
        )

    @classmethod
    def hinge_flip_uniform(cls):
        return cls(
            name="hinge_flip_uniform",
            allow_self_loops=True,
            allow_multiedges=True,
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
    def gibbs_uniform(cls):
        return cls(
            name="gibbs_uniform", sample_label_count_prob=0.1, label_creation_prob=0.5
        )

    @classmethod
    def gibbs_mixed(cls):
        return cls(
            name="gibbs_mixed",
            sample_label_count_prob=0.1,
            label_creation_prob=0.5,
            shift=1,
        )

    @classmethod
    def restricted_uniform(cls):
        return cls(name="restricted_uniform", sample_label_count_prob=0.1)

    @classmethod
    def restricted_mixed(cls):
        return cls(name="restricted_mixed", sample_label_count_prob=0.1, shift=1)

    @classmethod
    def nested_uniform(cls):
        return cls(name="nested_uniform", sample_label_count_prob=0.1)

    @classmethod
    def nested_mixed(cls):
        return cls(name="nested_mixed", sample_label_count_prob=0.1, shift=1)


class EdgeProposerFactory(Factory):
    @staticmethod
    def build_single_uniform(
        config: EdgeProposerConfig,
    ) -> proposer.edge.SingleEdgeUniformProposer:
        return proposer.edge.SingleEdgeUniformProposer(
            allow_self_loops=config.allow_self_loops,
            allow_multiedges=config.allow_multiedges,
        )

    @staticmethod
    def build_single_degree(
        config: EdgeProposerConfig,
    ) -> proposer.edge.SingleEdgeDegreeProposer:
        return proposer.edge.SingleEdgeDegreeProposer(
            config.shift,
            allow_self_loops=config.allow_self_loops,
            allow_multiedges=config.allow_multiedges,
        )

    @staticmethod
    def build_hinge_flip_uniform(
        config: EdgeProposerConfig,
    ) -> proposer.edge.HingeFlipUniformProposer:
        return proposer.edge.HingeFlipUniformProposer(
            allow_self_loops=config.allow_self_loops,
            allow_multiedges=config.allow_multiedges,
        )

    @staticmethod
    def build_hinge_flip_degree(
        config: EdgeProposerConfig,
    ) -> proposer.edge.HingeFlipDegreeProposer:
        return proposer.edge.HingeFlipDegreeProposer(
            config.shift,
            allow_self_loops=config.allow_self_loops,
            allow_multiedges=config.allow_multiedges,
        )

    @staticmethod
    def build_double_swap(
        config: EdgeProposerConfig,
    ) -> proposer.edge.DoubleEdgeSwapProposer:
        return proposer.edge.DoubleEdgeSwapProposer(
            allow_self_loops=config.allow_self_loops,
            allow_multiedges=config.allow_multiedges,
        )


class BlockProposerFactory(Factory):
    @staticmethod
    def build_gibbs_uniform(
        config: BlockProposerConfig,
    ) -> proposer.label.GibbsUniformBlockProposer:
        return proposer.label.GibbsUniformBlockProposer(
            config.sample_label_count_prob, config.label_creation_prob
        )

    @staticmethod
    def build_gibbs_mixed(
        config: BlockProposerConfig,
    ) -> proposer.label.GibbsMixedBlockProposer:
        return proposer.label.GibbsMixedBlockProposer(
            config.sample_label_count_prob, config.label_creation_prob, config.shift
        )

    @staticmethod
    def build_restricted_uniform(
        config: BlockProposerConfig,
    ) -> proposer.label.RestrictedUniformBlockProposer:
        return proposer.label.RestrictedUniformBlockProposer(
            config.sample_label_count_prob
        )

    @staticmethod
    def build_restricted_mixed(
        config: BlockProposerConfig,
    ) -> proposer.label.RestrictedMixedBlockProposer:
        return proposer.label.RestrictedMixedBlockProposer(
            config.sample_label_count_prob, config.shift
        )

    @staticmethod
    def build_nested_uniform(
        config: BlockProposerConfig,
    ) -> proposer.label.RestrictedUniformNestedBlockProposer:
        return proposer.label.RestrictedUniformNestedBlockProposer(
            config.sample_label_count_prob
        )

    @staticmethod
    def build_nested_mixed(
        config: BlockProposerConfig,
    ) -> proposer.label.RestrictedMixedNestedBlockProposer:
        return proposer.label.RestrictedMixedNestedBlockProposer(
            config.sample_label_count_prob, config.shift
        )
