import numpy as np
import typing

from typing import Union
from .config import Config
from .factory import Factory
from .prior import *
from _midynet import random_graph


class StochasticBlockModelConfig(Config):
    requirements: set[str] = {
        "name",
        "block_count",
        "blocks",
        "edge_count",
        "edge_matrix",
    }

    @classmethod
    def custom(
        cls,
        size: int,
        block_count: Union[int, float, tuple[int, int], BlockCountPriorConfig],
        blocks: Union[np.array, BlockPriorConfig],
        edge_count: Union[int, float, EdgeCountPriorConfig],
        edge_matrix: Union[np.array, EdgeMatrixPriorConfig],
    ):
        obj = cls(name="sbm", size=size)
        obj.insert("block_count", BlockCountPriorConfig.auto(block_count))
        obj.insert("blocks", BlockPriorConfig.auto(blocks))
        obj.insert("edge_count", EdgeCountPriorConfig.auto(edge_count))
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
        return cls.custom(size, block_count, blocks, edge_count, edge_matrix)

    @classmethod
    def uniform(cls, size: int, edge_count):
        block_count = BlockCountPriorConfig.uniform(size)
        blocks = BlockPriorConfig.uniform()
        edge_matrix = EdgeMatrixPriorConfig.uniform()
        return cls.custom(size, block_count, blocks, edge_count, edge_matrix)

    @classmethod
    def uniform_hyperprior(cls, size: int, edge_count):
        block_count = BlockCountPriorConfig.uniform(size)
        blocks = BlockPriorConfig.uniform_hyperprior()
        edge_matrix = EdgeMatrixPriorConfig.uniform()
        return cls.custom(size, block_count, blocks, edge_count, edge_matrix)


def sbm_builder(config: StochasticBlockModelConfig):
    return random_graph.StochasticBlockModelFamily(
        c.size, BlockPriorFactory.build(c), EdgeMatrixPriorFactory.build(c)
    )


class StochasticBlockModelFactory(Factory):
    options: Dict[str, Callable[[Config], random_graph.StochasticBlockModelFamily]] = {
        "custom": sbm_builder,
        "fixed": sbm_builder,
        "uniform": sbm_builder,
        "uniform_hyperprior": sbm_builder,
    }


if __name__ == "__main__":
    pass
