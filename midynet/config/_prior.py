import numpy as np
from basegraph.core import UndirectedMultigraph
from itertools import combinations_with_replacement
from _midynet.prior import sbm

from .config import Config
from .factory import Factory, UnavailableOption
from .wrapper import Wrapper

__all__ = (
    "EdgeCountPriorConfig",
    "BlockCountPriorConfig",
    "BlockPriorConfig",
    "EdgeMatrixPriorConfig",
    "DegreePriorConfig",
    "EdgeCountPriorFactory",
    "BlockCountPriorFactory",
    "BlockPriorFactory",
    "EdgeMatrixPriorFactory",
    "DegreePriorFactory",
)


class EdgeCountPriorConfig(Config):
    @classmethod
    def auto(cls, config):
        if isinstance(config, int):
            return cls.delta(config)
        elif isinstance(config, float):
            return cls.poisson(config)
        else:
            return super().auto(config)

    @classmethod
    def delta(cls, edge_count: int = 0):
        if edge_count < 0:
            raise ValueError("`edge_count` must be non-negative")
        return cls(name="delta", state=edge_count)

    @classmethod
    def poisson(cls, mean: float = 0):
        if mean < 0:
            raise ValueError("`mean` must be non-negative")
        return cls(name="poisson", mean=mean)


class BlockCountPriorConfig(Config):
    @classmethod
    def auto(cls, config):
        if isinstance(config, int):
            return cls.delta(config)
        elif isinstance(config, float):
            return cls.poisson(config)
        elif isinstance(config, tuple):
            min, max = config
            return cls.uniform(min, max)
        else:
            return super().auto(config)

    @classmethod
    def delta(cls, block_count: int):
        if block_count < 1:
            raise ValueError("`block_count` must be greater than or equal to 1.")
        return cls(name="delta", state=block_count)

    @classmethod
    def poisson(cls, mean: float = 1):
        if mean < 0:
            raise ValueError("`mean` must be non-negative.")
        return cls(name="poisson", mean=mean)

    @classmethod
    def uniform(cls, min: int = 1, max: int = None):
        if min is not None and max is None:
            min, max = 1, min
        if min < 1:
            raise ValueError("`min` must be greater than or equal to 1.")
        elif min > max:
            raise ValueError("`max` must be greater than or equal to `min`.")
        return cls(name="uniform", min=min, max=max)


class BlockPriorConfig(Config):
    @classmethod
    def auto(cls, config):
        if isinstance(config, list):
            return cls.delta(config)
        elif config == "uniform":
            return cls.uniform()
        elif config == "hyperuniform":
            return cls.hyperuniform()
        else:
            return super().auto(config)

    @classmethod
    def delta(cls, blocks: list[int]):
        return cls(name="delta", state=blocks)

    @classmethod
    def uniform(
        cls,
        size: int,
        bmin: int = 1,
        bmax: int = None,
    ):
        bmax = size if bmax is None else bmax
        return cls(
            name="uniform",
            size=size,
            block_count=BlockCountPriorConfig.uniform(bmin, bmax),
        )

    @classmethod
    def hyperuniform(
        cls,
        size: int,
        bmin: int = 1,
        bmax: int = None,
    ):
        bmax = size if bmax is None else bmax
        return cls(
            name="hyperuniform",
            size=size,
            block_count=BlockCountPriorConfig.uniform(bmin, bmax),
        )

    @classmethod
    def custom(cls, name: str, size: int, block_count: BlockCountPriorConfig):
        return cls(name=name, size=size, block_count=block_count)


class EdgeMatrixPriorConfig(Config):
    @classmethod
    def uniform(cls, edge_count: EdgeCountPriorConfig):
        return cls(name="uniform", edge_count=EdgeCountPriorConfig.auto(edge_count))

    @classmethod
    def delta(cls, edge_matrix: np.array):
        return cls(name="delta", state=edge_matrix)


class DegreePriorConfig(Config):
    @classmethod
    def auto(cls, config):
        if isinstance(config, list):
            return cls.delta(config)
        elif config == "uniform":
            return cls.uniform()
        elif config == "hyperuniform":
            return cls.hyperuniform()
        else:
            return super().auto(config)

    @classmethod
    def delta(cls, degrees: list[int]):
        return cls(name="delta", state=degrees)

    @classmethod
    def uniform(cls):
        return cls(name="uniform")

    @classmethod
    def hyperuniform(cls):
        return cls(name="hyperuniform")


class EdgeCountPriorFactory(Factory):
    @staticmethod
    def build_delta(config: EdgeCountPriorConfig) -> sbm.EdgeCountDeltaPrior:
        return sbm.EdgeCountDeltaPrior(config.get_value("state"))

    @staticmethod
    def build_poisson(config: EdgeCountPriorConfig) -> sbm.EdgeCountPoissonPrior:
        return sbm.EdgeCountPoissonPrior(config.get_value("mean"))


class BlockCountPriorFactory(Factory):
    @staticmethod
    def build_delta(config: BlockCountPriorConfig) -> sbm.BlockCountDeltaPrior:
        return sbm.BlockCountDeltaPrior(config.get_value("state", 1))

    @staticmethod
    def build_poisson(
        config: BlockCountPriorConfig,
    ) -> sbm.BlockCountPoissonPrior:
        return sbm.BlockCountPoissonPrior(config.get_value("mean", 1))

    @staticmethod
    def build_uniform(
        config: BlockCountPriorConfig,
    ) -> sbm.BlockCountUniformPrior:
        return sbm.BlockCountUniformPrior(
            config.get_value("min", 1), config.get_value("max", 1)
        )


class BlockPriorFactory(Factory):
    @staticmethod
    def build_delta(config: BlockPriorConfig) -> sbm.BlockDeltaPrior:
        b = sbm.BlockDeltaPrior()
        return b

    @staticmethod
    def build_uniform(config: BlockPriorConfig) -> sbm.BlockUniformPrior:
        if config.block_count.get_value("max") is None:
            config.block_count.set_value("max", config.get_value("size"))
        B = BlockCountPriorFactory.build(config.get_value("block_count"))
        b = sbm.BlockUniformPrior(config.get_value("size"), B)
        return Wrapper(
            b,
            setup_func=lambda wrap, others: wrap.set_block_count_prior(
                others["block_count"]
            ),
            block_count=B,
        )

    @staticmethod
    def build_hyperuniform(
        config: BlockPriorConfig,
    ) -> sbm.BlockUniformHyperPrior:
        if config.block_count.get_value("max") is None:
            config.block_count.set_value("max", config.get_value("size"))
        B = BlockCountPriorFactory.build(config.get_value("block_count"))
        b = sbm.BlockUniformHyperPrior(config.get_value("size"), B)

        return Wrapper(
            b,
            setup_func=lambda wrap, others: wrap.set_block_count_prior(
                others["block_count"]
            ),
            block_count=B,
        )


class EdgeMatrixPriorFactory(Factory):
    @staticmethod
    def build_uniform(config) -> sbm.EdgeMatrixUniformPrior:
        E = EdgeCountPriorFactory.build(config.edge_count)
        e = sbm.EdgeMatrixUniformPrior()
        e.set_edge_count_prior(E)
        return Wrapper(
            e,
            setup_func=lambda wrap, others: wrap.set_edge_count_prior(
                others["edge_count"]
            ),
            edge_count=E,
        )

    @staticmethod
    def build_delta(config) -> sbm.EdgeMatrixDeltaPrior:
        B = config.state.shape[0]
        e = UndirectedMultigraph(B)
        for r, s in combinations_with_replacement(range(B), r=2):
            e.set_edge_multiplicity_idx(r, s, config.state[r, s])
        return Wrapper(sbm.EdgeMatrixDeltaPrior(e), label_graph=e)


class DegreePriorFactory(Factory):
    @staticmethod
    def build_delta(config: DegreePriorConfig) -> sbm.DegreeDeltaPrior:
        return sbm.DegreeDeltaPrior()

    @staticmethod
    def build_uniform(config: DegreePriorConfig) -> sbm.DegreeUniformPrior:
        return sbm.DegreeUniformPrior()

    @staticmethod
    def build_hyperuniform(config: DegreePriorConfig) -> sbm.DegreePrior:
        return sbm.DegreeUniformHyperPrior()


if __name__ == "__main__":
    pass
