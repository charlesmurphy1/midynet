import numpy as np
import typing

from typing import Union
from .config import Config
from .factory import Factory
from .wrapper import Wrapper
from .prior import *
from .proposer import *
from midynet.util.degree_sequences import *
from _midynet import random_graph
from _midynet.prior import sbm

__all__ = ["RandomGraphConfig", "RandomGraphFactory"]


class RandomGraphConfig(Config):
    requirements: set[str] = {"size"}

    @classmethod
    def custom_sbm(
        cls,
        name: str,
        size: int,
        blocks: Union[np.array, BlockPriorConfig],
        edge_matrix: Union[np.array, EdgeMatrixPriorConfig],
    ):
        obj = cls(name=name, size=size)
        obj.insert("blocks", BlockPriorConfig.auto(blocks))
        obj.insert("edge_matrix", EdgeMatrixPriorConfig.auto(edge_matrix))
        if obj.edge_matrix.edge_count.name == "delta":
            obj.insert("edge_proposer", EdgeProposerConfig.hinge_flip_uniform())
        else:
            obj.insert("edge_proposer", EdgeProposerConfig.single_uniform())
        obj.insert("block_proposer", BlockProposerConfig.peixoto())
        return obj

    @classmethod
    def uniform_sbm(
        cls, size: int = 100, edge_count: int = 250, block_count_max: int = None
    ):
        blocks = BlockPriorConfig.uniform()
        blocks = BlockPriorConfig.uniform()
        blocks.block_count.set_value("max", block_count_max)
        edge_matrix = EdgeMatrixPriorConfig.uniform(edge_count)
        return cls.custom_sbm("uniform_sbm", size, blocks, edge_matrix)

    @classmethod
    def hyperuniform_sbm(
        cls, size: int = 100, edge_count: int = 250, block_count_max: int = None
    ):
        blocks = BlockPriorConfig.hyperuniform()
        edge_matrix = EdgeMatrixPriorConfig.uniform(edge_count)
        return cls.custom_sbm("hyperuniform_sbm", size, blocks, edge_matrix)

    def planted_partition(
        cls, size: int, edge_count: int, block_count: int, assortativity: float
    ):
        obj = cls(
            "planted_partition",
            size=size,
            edge_count=edge_count,
            block_count=block_count,
            assortativity=assortativity,
        )
        obj.insert("blocks", BlockPriorConfig.delta(block_count))
        obj.insert("edge_matrix", EdgeMatrixPriorConfig.uniform(edge_count))
        obj.insert("edge_proposer", EdgeProposerConfig.hinge_flip_uniform())
        obj.insert("block_proposer", BlockProposerConfig.peixoto())

    @classmethod
    def custom_er(cls, name: str, size: int, edge_count: EdgeCountPriorConfig):
        obj = cls(name=name, size=size)
        obj.insert("edge_count", EdgeCountPriorConfig.auto(edge_count))

        if obj.edge_count.name == "delta":
            obj.insert("edge_proposer", EdgeProposerConfig.hinge_flip_uniform())
        else:
            obj.insert("edge_proposer", EdgeProposerConfig.single_uniform())
        return obj

    @classmethod
    def er(cls, size: int = 100, edge_count: int = 250):
        edge_count = EdgeCountPriorConfig.auto(edge_count)
        return cls.custom_er(name="er", size=size, edge_count=edge_count)

    @classmethod
    def custom_dcsbm(
        cls,
        name: str,
        size: int,
        blocks: Union[np.array, BlockPriorConfig],
        edge_matrix: Union[np.array, EdgeMatrixPriorConfig],
        degrees: Union[np.array, DegreePriorConfig],
    ):
        obj = cls(name=name, size=size)
        obj.insert("blocks", BlockPriorConfig.auto(blocks))
        obj.insert("edge_matrix", EdgeMatrixPriorConfig.auto(edge_matrix))
        obj.insert("degrees", DegreePriorConfig.auto(degrees))
        if obj.degrees.name == "delta":
            obj.insert("edge_proposer", EdgeProposerConfig.double_swap())
        elif obj.edge_matrix.edge_count.name == "delta":
            obj.insert("edge_proposer", EdgeProposerConfig.hinge_flip_uniform())
        else:
            obj.insert("edge_proposer", EdgeProposerConfig.single_uniform())
        obj.insert("block_proposer", BlockProposerConfig.peixoto())
        return obj

    @classmethod
    def uniform_dcsbm(
        cls, size: int = 100, edge_count: int = 250, block_count_max: int = None
    ):
        blocks = BlockPriorConfig.uniform()
        blocks.block_count.set_value("max", block_count_max)
        edge_matrix = EdgeMatrixPriorConfig.uniform(edge_count)
        degrees = DegreePriorConfig.uniform()

        return cls.custom_dcsbm("uniform_dcsbm", size, blocks, edge_matrix, degrees)

    @classmethod
    def hyperuniform_dcsbm(
        cls, size: int = 100, edge_count: int = 250, block_count_max: int = None
    ):
        blocks = BlockPriorConfig.hyperuniform()
        blocks.block_count.set_value("max", block_count_max)
        edge_matrix = EdgeMatrixPriorConfig.uniform(edge_count)
        degrees = DegreePriorConfig.hyperuniform()
        return cls.custom_dcsbm(
            "hyperuniform_dcsbm", size, blocks, edge_matrix, degrees
        )

    @classmethod
    def custom_cm(
        cls,
        name: str,
        size: int,
        edge_count: EdgeCountPriorConfig,
        degrees: DegreePriorConfig,
    ):
        obj = cls(name=name, size=size)
        obj.insert("edge_count", EdgeCountPriorConfig.auto(edge_count))
        obj.insert("degrees", DegreePriorConfig.auto(degrees))
        if obj.degrees.name == "delta":
            obj.insert("edge_proposer", EdgeProposerConfig.double_swap())
        elif obj.edge_count.name == "delta":
            obj.insert("edge_proposer", EdgeProposerConfig.hinge_flip_uniform())
        else:
            obj.insert("edge_proposer", EdgeProposerConfig.single_uniform())
        return obj

    @classmethod
    def poisson_cm(cls, size: int = 100, edge_count: int = 250):
        obj = cls("poisson_cm", size=size, edge_count=edge_count)
        obj.insert("edge_proposer", EdgeProposerConfig.double_swap())
        return obj

    @classmethod
    def nbinom_cm(cls, size: int = 100, edge_count: int = 250, heterogeneity: int = 0):
        obj = cls(
            "nbinom_cm",
            size=size,
            edge_count=edge_count,
            heterogeneity=heterogeneity,
        )
        obj.insert("edge_proposer", EdgeProposerConfig.double_swap())
        return obj

    @classmethod
    def uniform_cm(cls, size: int = 100, edge_count: int = 250):
        edge_count = EdgeCountPriorConfig.auto(edge_count)
        degrees = DegreePriorConfig.uniform()
        return cls.custom_cm("uniform_cm", size, edge_count, degrees)

    @classmethod
    def hyperuniform_cm(cls, size: int = 100, edge_count: int = 250):
        edge_count = EdgeCountPriorConfig.auto(edge_count)
        degrees = DegreePriorConfig.hyperuniform()
        return cls.custom_cm("hyperuniform_cm", size, edge_count, degrees)


class RandomGraphFactory(Factory):
    @staticmethod
    def setUpSBM(graph, blocks, edge_matrix):
        graph.set_block_prior(blocks)
        graph.set_edge_matrix_prior(edge_matrix)

    @staticmethod
    def setUpER(graph, edge_count):
        graph.set_edge_count_prior(edge_count)

    @staticmethod
    def setUpDCSBM(graph, blocks, edge_matrix, degrees):
        graph.set_block_prior(blocks)
        graph.set_edge_matrix_prior(edge_matrix)
        graph.set_degree_prior(degrees)

    @staticmethod
    def setUpCM(graph, edge_count, degrees):
        graph.set_edge_count_prior(edge_count)
        graph.set_degree_prior(degrees)

    @staticmethod
    def build_custom_sbm(
        config: RandomGraphConfig,
    ) -> random_graph.StochasticBlockModelFamily:
        block_wrapper = BlockPriorFactory.build(config.blocks)
        block_wrapper.set_size(config.size)
        edge_matrix_wrapper = EdgeMatrixPriorFactory.build(config.edge_matrix)

        g = random_graph.StochasticBlockModelFamily(config.size)
        return Wrapper(
            g,
            setup_func=lambda wrap, others: RandomGraphFactory.setUpSBM(
                wrap,
                others["blocks"].get_wrap(),
                others["edge_matrix"].get_wrap(),
            ),
            blocks=block_wrapper,
            edge_matrix=edge_matrix_wrapper,
        )

    @staticmethod
    def build_custom_fixed_sbm(
        blocks: list[int], edge_matrix: list[list[int]]
    ) -> random_graph.StochasticBlockModelFamily:
        # block_prior = sbm.BlockDeltaPrior(blocks)
        # edge_matrix_prior = sbm.EdgeMatrixUniformPrior()
        #
        # g = random_graph.StochasticBlockModelFamily(config.size)
        # return Wrapper(
        #     g,
        #     setup_func=lambda wrap, others: RandomGraphFactory.setUpSBM(
        #         wrap,
        #         others["blocks"].get_wrap(),
        #         others["edge_matrix"].get_wrap(),
        #     ),
        #     blocks=block_wrapper,
        #     edge_matrix=edge_matrix_wrapper,
        # )
        UnavailableOption("fixed_sbm")

    @staticmethod
    def build_uniform_sbm(config: RandomGraphConfig):
        config.blocks.block_count.max = (
            config.size
            if config.blocks.block_count.max is None
            else config.blocks.block_count.max
        )
        return RandomGraphFactory.build_custom_sbm(config)

    @staticmethod
    def build_hyperuniform_sbm(config: RandomGraphConfig):
        config.blocks.block_count.max = (
            config.size
            if config.blocks.block_count.max is None
            else config.blocks.block_count.max
        )
        return RandomGraphFactory.build_custom_sbm(config)

    @staticmethod
    def build_planted_partition(config: RandomGraphConfig):
        UnavailableOption(config.name)

    @staticmethod
    def build_custom_er(config: RandomGraphConfig):
        edge_count = EdgeCountPriorFactory.build(config.edge_count)
        g = random_graph.ErdosRenyiFamily(config.size, edge_count)

        return Wrapper(
            g,
            setup_func=lambda wrap, others: RandomGraphFactory.setUpER(
                wrap,
                others["edge_count"],
            ),
            edge_count=edge_count,
        )

    @staticmethod
    def build_er(config: RandomGraphConfig):
        return RandomGraphFactory.build_custom_er(config)

    @staticmethod
    def build_custom_dcsbm(
        config: RandomGraphConfig,
    ) -> random_graph.DegreeCorrectedStochasticBlockModelFamily:
        block_wrapper = BlockPriorFactory.build(config.blocks)
        block_wrapper.set_size(config.size)
        edge_matrix_wrapper = EdgeMatrixPriorFactory.build(config.edge_matrix)
        degrees = DegreePriorFactory.build(config.degrees)
        g = random_graph.DegreeCorrectedStochasticBlockModelFamily(config.size)
        return Wrapper(
            g,
            setup_func=lambda wrap, others: RandomGraphFactory.setUpDCSBM(
                wrap,
                others["blocks"].get_wrap(),
                others["edge_matrix"].get_wrap(),
                others["degrees"],
            ),
            blocks=block_wrapper,
            edge_matrix=edge_matrix_wrapper,
            degrees=degrees,
        )

    @staticmethod
    def build_uniform_dcsbm(config: RandomGraphConfig):
        config.blocks.block_count.max = (
            config.size
            if config.blocks.block_count.max is None
            else config.blocks.block_count.max
        )
        return RandomGraphFactory.build_custom_dcsbm(config)

    @staticmethod
    def build_hyperuniform_dcsbm(config: RandomGraphConfig):
        config.blocks.block_count.max = (
            config.size
            if config.blocks.block_count.max is None
            else config.blocks.block_count.max
        )
        return RandomGraphFactory.build_custom_dcsbm(config)

    @staticmethod
    def build_custom_cm(
        config: RandomGraphConfig,
    ) -> random_graph.ConfigurationModelFamily:
        edge_count = EdgeCountPriorFactory.build(config.edge_count)
        degrees = DegreePriorFactory.build(config.degrees)
        g = random_graph.ConfigurationModelFamily(config.size)
        return Wrapper(
            g,
            setup_func=lambda wrap, others: RandomGraphFactory.setUpCM(
                wrap,
                others["edge_count"],
                others["degrees"],
            ),
            edge_count=edge_count,
            degrees=degrees,
        )

    @staticmethod
    def build_fixed_custom_cm(
        degrees: list[int],
    ) -> random_graph.ConfigurationModelFamily:
        size, edge_count = len(degrees), int(sum(degrees) / 2)
        print(sum(degrees))
        degree_prior = sbm.DegreeDeltaPrior(degrees)
        edge_count_prior = sbm.EdgeCountDeltaPrior(edge_count)
        g = random_graph.ConfigurationModelFamily(size)
        return Wrapper(
            g,
            setup_func=lambda wrap, others: RandomGraphFactory.setUpCM(
                wrap,
                others["edge_count"],
                others["degrees"],
            ),
            edge_count=edge_count_prior,
            degrees=degree_prior,
        )

    @staticmethod
    def build_poisson_cm(config: RandomGraphConfig):
        degrees = poisson_degreeseq(config.size, 2 * config.edge_count / config.size)
        print("poisson")
        return RandomGraphFactory.build_fixed_custom_cm(degrees)

    @staticmethod
    def build_nbinom_cm(config: RandomGraphConfig):
        degrees = nbinom_degreeseq(
            config.size, 2 * config.edge_count / config.size, config.heterogeneity
        )
        print("nbinom")
        return RandomGraphFactory.build_fixed_custom_cm(degrees)

    @staticmethod
    def build_uniform_cm(config: RandomGraphConfig):
        return RandomGraphFactory.build_custom_cm(config)

    @staticmethod
    def build_hyperuniform_cm(config: RandomGraphConfig):
        return RandomGraphFactory.build_custom_cm(config)


if __name__ == "__main__":
    pass
