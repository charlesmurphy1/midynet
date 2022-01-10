import numpy as np
import typing

from typing import Union
from .config import Config
from .factory import Factory
from .prior import *
from _midynet import random_graph


class StochasticBlockModelConfig(Config):
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
    def uniform(cls, size: int = 100, edge_count: int = 250):
        blocks = BlockPriorConfig.uniform()
        blocks.block_count.set_value("max", size)
        edge_matrix = EdgeMatrixPriorConfig.uniform(edge_count)
        return cls.custom("uniform", size, blocks, edge_matrix)

    @classmethod
    def hyperuniform(cls, size: int = 100, edge_count: int = 250):
        blocks = BlockPriorConfig.hyperuniform()
        blocks.block_count.set_value("max", size)
        edge_matrix = EdgeMatrixPriorConfig.uniform(edge_count)
        return cls.custom("hyperuniform", size, blocks, edge_matrix)


def sbm_builder(config: StochasticBlockModelConfig):
    block_count = BlockCountPriorFactory.build(config.blocks.block_count)
    blocks = BlockPriorFactory.build(config.blocks)
    blocks.set_size(config.size)
    blocks.set_block_count_prior(block_count)

    edge_count = EdgeCountPriorFactory.build(config.edge_matrix.edge_count)
    edge_matrix = EdgeMatrixPriorFactory.build(config.edge_matrix)
    edge_matrix.set_edge_count_prior(edge_count)
    edge_matrix.set_block_prior(blocks)

    g = random_graph.StochasticBlockModelFamily(config.size, blocks, edge_matrix)

    print("sampling block_count", block_count.sample())
    print("sampling blocks", blocks.sample())
    print("sampling edge_count", edge_count.sample())
    print("sampling edge_matrix", edge_matrix.sample())
    # print("sampling graph", g.sample())

    return g


class StochasticBlockModelFactory(Factory):
    options: Dict[str, Callable[[Config], random_graph.StochasticBlockModelFamily]] = {
        "custom": sbm_builder,
        "fixed": sbm_builder,
        "uniform": sbm_builder,
        "hyperuniform": sbm_builder,
    }


if __name__ == "__main__":
    pass
