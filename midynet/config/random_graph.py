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
        block_count = max(blocks) + 1
        edge_count = sum(edge_matrix) / 2
        if edge_matrix != (block_count, block_count):
            message = (
                f"Invalid shape {edge_matrix.shape} for `edge_matrix`,"
                + f"expected shape {(block_count, block_count)}"
            )
            raise ValueError(message)
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


class StochasticBlockModelFamilyFactory(Factory):
    @staticmethod
    def setUpSBM(sbm_graph, blocks, edge_matrix):
        sbm_graph.set_block_prior(blocks)
        sbm_graph.set_edge_matrix_prior(edge_matrix)

    @staticmethod
    def build_custom(
        config: StochasticBlockModelFamilyConfig,
    ) -> random_graph.StochasticBlockModelFamily:
        block_wrapper = BlockPriorFactory.build(config.blocks)
        block_wrapper.set_size(config.size)

        edge_matrix_wrapper = EdgeMatrixPriorFactory.build(config.edge_matrix)
        edge_matrix_wrapper.set_block_prior(block_wrapper.get_wrapped())

        g = random_graph.StochasticBlockModelFamily(config.size)
        g.set_block_prior(block_wrapper.get_wrapped())
        g.set_edge_matrix_prior(edge_matrix_wrapper.get_wrapped())
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


if __name__ == "__main__":
    pass
