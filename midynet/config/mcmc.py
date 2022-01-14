from __future__ import annotations

from .config import Config
from .random_graph import RandomGraphConfig
from .factory import Factory
from .wrapper import Wrapper
from _midynet import proposer
from _midynet import mcmc

__all__ = [
    "EdgeProposerConfig",
    "BlockProposerConfig",
    "MCMCConfig",
    "EdgeProposerFactory",
    "BlockProposerFactory",
    "RandomGraphMCMCFactory",
    "DynamicsMCMCFactory",
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


class MCMCConfig(Config):
    @classmethod
    def sbm(cls, config: RandomGraphConfig, block_proposer_name="peixoto"):
        if config.edge_matrix.edge_count.name == "delta":
            edge_proposer = EdgeProposerConfig.hinge_flip_degree()
        else:
            edge_proposer = EdgeProposerConfig.single_degree()
        block_proposer = BlockProposerConfig.auto(block_proposer_name)
        if config.blocks.block_count.name == "delta":
            block_proposer.set_value("create_new_block", 0)

        return cls(
            name="sbm",
            edge_proposer=edge_proposer,
            block_proposer=block_proposer,
        )

    @classmethod
    def er(cls, config: RandomGraphConfig):
        if config.edge_count.name == "delta":
            edge_proposer = EdgeProposerConfig.hinge_flip_degree()
        else:
            edge_proposer = EdgeProposerConfig.single_degree()
        return cls(name="er", edge_proposer=edge_proposer)

    @classmethod
    def dcsbm(cls, config: RandomGraphConfig, block_proposer_name="peixoto"):
        if config.degrees.name == "delta":
            edge_proposer = EdgeProposerConfig.double_swap()
        elif config.edge_matrix.edge_count.name == "delta":
            edge_proposer = EdgeProposerConfig.hinge_flip_degree()
        else:
            edge_proposer = EdgeProposerConfig.single_degree()
        block_proposer = BlockProposerConfig.auto(block_proposer_name)
        if config.blocks.block_count.name == "delta":
            block_proposer.set_value("create_new_block", 0)

        return cls(
            name="dcsbm",
            edge_proposer=edge_proposer,
            block_proposer=block_proposer,
        )

    @classmethod
    def cm(cls, config: RandomGraphConfig):
        if config.degrees.name == "delta":
            edge_proposer = EdgeProposerConfig.double_swap()
        elif config.edge_count.name == "delta":
            edge_proposer = EdgeProposerConfig.hinge_flip_degree()
        else:
            edge_proposer = EdgeProposerConfig.single_degree()

        return cls(name="cm", edge_proposer=edge_proposer)


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


class RandomGraphMCMCFactory(Factory):
    @staticmethod
    def setup_sbm_mcmc(mcmc_model, block_proposer):
        mcmc_model.set_block_proposer(block_proposer)

    @staticmethod
    def build_sbm(config: MCMCConfig) -> mcmc.StochasticBlockGraphMCMC:
        block_proposer = BlockProposerFactory.build(config.block_proposer)
        mcmc_model = mcmc.StochasticBlockGraphMCMC()
        return Wrapper(
            mcmc_model,
            setup_func=lambda wrap, others: RandomGraphMCMCFactory.setup_sbm_mcmc(
                wrap, others["block_proposer"]
            ),
            block_proposer=block_proposer,
        )

    @staticmethod
    def build_er(config: MCMCConfig) -> mcmc.RandomGraphMCMC:
        return Wrapper(mcmc.RandomGraphMCMC())

    @staticmethod
    def build_dcsbm(config: MCMCConfig) -> mcmc.StochasticBlockGraphMCMC:
        return RandomGraphMCMCFactory.build_sbm(config)

    @staticmethod
    def build_cm(config: MCMCConfig) -> mcmc.RandomGraphMCMC:
        return RandomGraphMCMCFactory.build_er(config)


class DynamicsMCMCFactory(Factory):
    @staticmethod
    def setup_erlike(dynamics_mcmc_model, graph_mcmc_model, edge_proposer):
        dynamics_mcmc_model.set_edge_proposer(edge_proposer)
        dynamics_mcmc_model.set_random_graph_mcmc(graph_mcmc_model)

    @staticmethod
    def setup_sbmlike(
        dynamics_mcmc_model, graph_mcmc_model, edge_proposer, block_proposer
    ):
        graph_mcmc_model.set_block_proposer(block_proposer)
        dynamics_mcmc_model.set_edge_proposer(edge_proposer)
        dynamics_mcmc_model.set_random_graph_mcmc(graph_mcmc_model)

    @staticmethod
    def build_sbm(config: MCMCConfig) -> mcmc.DynamicsMCMC:
        edge_proposer = EdgeProposerFactory.build(config.edge_proposer)
        block_proposer = BlockProposerFactory.build(config.block_proposer)
        graph_mcmc_model = mcmc.StochasticBlockGraphMCMC()
        dynamics_mcmc_model = mcmc.DynamicsMCMC()
        return Wrapper(
            dynamics_mcmc_model,
            setup_func=lambda wrap, others: DynamicsMCMCFactory.setup_sbmlike(
                wrap,
                others["graph_mcmc_model"],
                others["edge_proposer"],
                others["block_proposer"],
            ),
            graph_mcmc_model=graph_mcmc_model,
            edge_proposer=edge_proposer,
            block_proposer=block_proposer,
        )

    @staticmethod
    def build_er(config: MCMCConfig) -> mcmc.DynamicsMCMC:
        edge_proposer = EdgeProposerFactory.build(config.edge_proposer)
        graph_mcmc_model = mcmc.RandomGraphMCMC()
        dynamics_mcmc_model = mcmc.DynamicsMCMC()
        return Wrapper(
            dynamics_mcmc_model,
            setup_func=lambda wrap, others: DynamicsMCMCFactory.setup_erlike(
                wrap,
                others["graph_mcmc_model"],
                others["edge_proposer"],
            ),
            graph_mcmc_model=graph_mcmc_model,
            edge_proposer=edge_proposer,
        )

    @staticmethod
    def build_dcsbm(config: MCMCConfig) -> mcmc.DynamicsMCMC:
        return DynamicsMCMCFactory.build_sbm(config)

    @staticmethod
    def build_cm(config: MCMCConfig) -> mcmc.DynamicsMCMC:
        return DynamicsMCMCFactory.build_er(config)


if __name__ == "__main__":
    pass
