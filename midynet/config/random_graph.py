import numpy as np
import typing

from typing import Union
from .config import Config
from .factory import Factory
from .wrapper import Wrapper
from .prior import *
from _midynet import random_graph


class StochasticBlockModelFamilyConfig(Config):
    requirements: set[str] = {"name", "size", "blocks", "edge_matrix"}

    @classmethod
    def custom(
        cls,
        name: str,
        size: int,
        blocks: Union[np.array, BlockPriorConfig],
        edge_matrix: Union[np.array, EdgeMatrixPriorConfig],
    ):
        obj = cls(name=name, size=size)
        obj.insert("blocks", BlockPriorConfig.auto(blocks))
        obj.insert("edge_matrix", EdgeMatrixPriorConfig.auto(edge_matrix))
        return obj

    @classmethod
    def fixed(cls, blocks, edge_matrix):
        size = len(blocks)
        return cls.custom("fixed", size, blocks, edge_matrix)

    @classmethod
    def uniform(
        cls, size: int = 100, edge_count: int = 250, block_count_max: int = None
    ):
        blocks = BlockPriorConfig.uniform(size=size, block_count_max=block_count_max)
        edge_matrix = EdgeMatrixPriorConfig.uniform(edge_count)
        return cls.custom("uniform", size, blocks, edge_matrix)

    @classmethod
    def hyperuniform(
        cls, size: int = 100, edge_count: int = 250, block_count_max: int = None
    ):
        blocks = BlockPriorConfig.hyperuniform(
            size=size, block_count_max=block_count_max
        )
        blocks.block_count.set_value("max", size)
        edge_matrix = EdgeMatrixPriorConfig.uniform(edge_count)
        return cls.custom("hyperuniform", size, blocks, edge_matrix)


class ErdosRenyiFamilyConfig(Config):
    requirements: set[str] = {"name", "size", "edge_count"}

    @classmethod
    def custom(cls, name: str, size: int, edge_count: EdgeCountPriorConfig):
        obj = cls(name="custom", size=size)
        obj.insert("edge_count", EdgeCountPriorConfig.auto(edge_count))
        return obj

    @classmethod
    def fixed(cls, size: int = 100, edge_count: int = 250):
        edge_count = EdgeCountPriorConfig.delta(edge_count)
        return cls.custom(name="fixed", size=size, edge_count=edge_count)

    @classmethod
    def poisson(cls, size: int = 100, mean: int = 250):
        edge_count = EdgeCountPriorConfig.poisson(mean)
        return cls.custom(name="poisson", size=size, edge_count=edge_count)


class DegreeCorrectedStochasticBlockModelFamilyConfig(Config):
    requirements: set[str] = {"name", "size", "blocks", "edge_matrix", "degrees"}

    @classmethod
    def custom(
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
        return obj

    @classmethod
    def fixed(cls, blocks, edge_matrix, degrees):
        size = len(blocks)
        return cls.custom("fixed", size, blocks, edge_matrix, degrees)

    @classmethod
    def uniform(
        cls, size: int = 100, edge_count: int = 250, block_count_max: int = None
    ):
        blocks = BlockPriorConfig.uniform(size=size, block_count_max=block_count_max)
        edge_matrix = EdgeMatrixPriorConfig.uniform(edge_count)
        degrees = DegreePriorConfig.uniform()
        return cls.custom("uniform", size, blocks, edge_matrix, degrees)

    @classmethod
    def hyperuniform(
        cls, size: int = 100, edge_count: int = 250, block_count_max: int = None
    ):
        blocks = BlockPriorConfig.hyperuniform(
            size=size, block_count_max=block_count_max
        )
        edge_matrix = EdgeMatrixPriorConfig.uniform(edge_count)
        degrees = DegreePriorConfig.hyperuniform()
        return cls.custom("hyperuniform", size, blocks, edge_matrix, degrees)


class ConfigurationModelFamilyConfig(Config):
    pass


class StochasticBlockModelFamilyFactory(Factory):
    @staticmethod
    def setUpSBM(graph, blocks, edge_matrix):
        graph.set_block_prior(blocks)
        graph.set_edge_matrix_prior(edge_matrix)

    @staticmethod
    def build_custom(
        config: StochasticBlockModelFamilyConfig,
    ) -> random_graph.StochasticBlockModelFamily:
        block_wrapper = BlockPriorFactory.build(config.blocks)
        edge_matrix_wrapper = EdgeMatrixPriorFactory.build(config.edge_matrix)

        g = random_graph.StochasticBlockModelFamily(config.size)
        return Wrapper(
            g,
            setup_func=lambda wrap, others: StochasticBlockModelFamilyFactory.setUpSBM(
                wrap,
                others["blocks"].get_wrapped(),
                others["edge_matrix"].get_wrapped(),
            ),
            blocks=block_wrapper,
            edge_matrix=edge_matrix_wrapper,
        )

    @staticmethod
    def build_fixed(config: StochasticBlockModelFamilyConfig):
        return StochasticBlockModelFamilyFactory.build_custom(config)

    @staticmethod
    def build_uniform(config: StochasticBlockModelFamilyConfig):
        return StochasticBlockModelFamilyFactory.build_custom(config)

    @staticmethod
    def build_hyperuniform(config: StochasticBlockModelFamilyConfig):
        return StochasticBlockModelFamilyFactory.build_custom(config)


class ErdosRenyiFamilyFactory(Factory):
    @staticmethod
    def setUpER(graph, edge_count):
        graph.set_edge_count_prior(edge_count)

    @staticmethod
    def build_custom(config: ErdosRenyiFamilyConfig):
        edge_count = EdgeCountPriorFactory.build(config.edge_count)
        g = random_graph.ErdosRenyiFamily(config.size, edge_count)

        return Wrapper(
            g,
            setup_func=lambda wrap, others: ErdosRenyiFamilyFactory.setUpER(
                wrap,
                others["edge_count"],
            ),
            edge_count=edge_count,
        )

    @staticmethod
    def build_fixed(config: ErdosRenyiFamilyConfig):
        return ErdosRenyiFamilyFactory.build_custom(config)

    @staticmethod
    def build_poisson(config: ErdosRenyiFamilyConfig):
        return ErdosRenyiFamilyFactory.build_custom(config)


class DegreeCorrectedStochasticBlockModelFamilyFactory(Factory):
    @staticmethod
    def setUpDCSBM(graph, blocks, edge_matrix, degrees):
        graph.set_block_prior(blocks)
        graph.set_edge_matrix_prior(edge_matrix)
        graph.set_degree_prior(degrees)

    @staticmethod
    def build_custom(
        config: DegreeCorrectedStochasticBlockModelFamilyConfig,
    ) -> random_graph.DegreeCorrectedStochasticBlockModelFamily:
        block_wrapper = BlockPriorFactory.build(config.blocks)
        edge_matrix_wrapper = EdgeMatrixPriorFactory.build(config.edge_matrix)
        degrees = DegreePriorFactory.build(config.degrees)
        g = random_graph.DegreeCorrectedStochasticBlockModelFamily(config.size)
        return Wrapper(
            g,
            setup_func=lambda wrap, others: DegreeCorrectedStochasticBlockModelFamilyFactory.setUpDCSBM(
                wrap,
                others["blocks"].get_wrapped(),
                others["edge_matrix"].get_wrapped(),
                others["degrees"],
            ),
            blocks=block_wrapper,
            edge_matrix=edge_matrix_wrapper,
            degrees=degrees,
        )

    @staticmethod
    def build_fixed(config: DegreeCorrectedStochasticBlockModelFamilyConfig):
        return DegreeCorrectedStochasticBlockModelFamilyFactory.build_custom(config)

    @staticmethod
    def build_uniform(config: DegreeCorrectedStochasticBlockModelFamilyConfig):
        return DegreeCorrectedStochasticBlockModelFamilyFactory.build_custom(config)

    @staticmethod
    def build_hyperuniform(config: DegreeCorrectedStochasticBlockModelFamilyConfig):
        return DegreeCorrectedStochasticBlockModelFamilyFactory.build_custom(config)


class ConfigurationModelFamilyFactory(Factory):
    pass


if __name__ == "__main__":
    pass
