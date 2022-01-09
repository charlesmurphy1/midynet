from typing import Any, Callable, Dict

from .config import Config
from .factory import Factory, UnavailableOption
from _midynet import prior


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
    def delta(cls, edge_count: int):
        if edge_count < 0:
            raise ValueError(f"`edge_count` must be non-negative")
        return cls(name="delta", state=edge_count)

    @classmethod
    def poisson(cls, mean: float):
        if mean < 0:
            raise ValueError(f"`mean` must be non-negative")
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
            raise ValueError(f"`block_count` must be greater than or equal to 1.")
        return cls(name="delta", state=block_count)

    @classmethod
    def poisson(cls, mean: float):
        if mean < 0:
            raise ValueError(f"`mean` must be non-negative.")
        return cls(name="poisson", mean=mean)

    @classmethod
    def uniform(cls, min: int, max: int = None):
        if max is None:
            max, min = min, 1
        if min < 1:
            raise ValueError(f"`min` must be greater than or equal to 1.")
        elif min > max:
            raise ValueError(f"`max` must be greater than or equal to `min`.")
        return cls(name="uniform", min=min, max=max)


class BlockPriorConfig(Config):
    @classmethod
    def auto(cls, config):
        if isinstance(config, list):
            return cls.delta(config)
        elif config == "uniform":
            return cls.uniform()
        elif config == "uniform_hyperprior":
            return cls.uniform_hyperprior()
        else:
            return super().auto(config)

    @classmethod
    def delta(cls, blocks: list[int]):
        return cls(name="delta", state=blocks)

    @classmethod
    def uniform(cls):
        return cls(name="uniform")

    @classmethod
    def uniform_hyperprior(cls):
        return cls(name="uniform_hyperprior")


class EdgeMatrixPriorConfig(Config):
    @classmethod
    def uniform(cls):
        return cls(name="uniform")


class DegreePriorConfig(Config):
    @classmethod
    def auto(cls, config):
        if isinstance(config, list):
            return cls.delta(config)
        elif config == "uniform":
            return cls.uniform()
        elif config == "uniform_hyperprior":
            return cls.uniform_hyperprior()
        else:
            return super().auto(config)

    @classmethod
    def delta(cls, degrees: list[int]):
        return cls(name="delta", state=degrees)

    @classmethod
    def uniform(cls):
        return cls(name="uniform")

    @classmethod
    def uniform_hyperprior(cls):
        return cls(name="uniform_hyperprior")


class EdgeCountPriorFactory(Factory):
    options: Dict[str, Callable[[Config], prior.sbm.EdgeCountPrior]] = {
        "delta": lambda c: prior.sbm.EdgeCountDeltaPrior(c.edge_count.state),
        "poisson": lambda c: prior.sbm.EdgeCountPoissonPrior(c.edge_count.mean),
    }


class BlockCountPriorFactory(Factory):
    options: Dict[str, Callable[[Config], prior.sbm.BlockCountPrior]] = {
        "delta": lambda c: prior.sbm.BlockCountDeltaPrior(c.block_count.state),
        "poisson": lambda c: prior.sbm.BlockCountPoissonPrior(c.block_count.mean),
        "uniform": lambda c: UnavailableOption(c.block_count.name),
    }


class BlockPriorFactory(Factory):
    options: Dict[str, Callable[[Config], prior.sbm.BlockPrior]] = {
        "uniform": lambda c: prior.sbm.BlockUniformPrior(
            c.size, BlockCountPriorFactory.build(c, "block_count")
        ),
        "uniform_hyperprior": lambda c: prior.sbm.BlockUniformHyperPrior(
            c.size, BlockCountPriorFactory.build(c, "block_count")
        ),
    }


class EdgeMatrixPriorFactory(Factory):
    options: Dict[str, Callable[[Config], prior.sbm.EdgeMatrixPrior]] = {
        "uniform": lambda c: prior.sbm.EdgeMatrixUniformPrior(
            EdgeCountPriorFactory.build(c, "edge_count"),
            BlocksPriorFactory.build(c, "blocks"),
        ),
    }


class DegreePriorFactory(Factory):
    options: Dict[str, Callable[[Config], prior.sbm.DegreePrior]] = {
        "uniform": lambda c: prior.sbm.DegreeUniformPrior(
            BlocksPriorFactory.build(c, "blocks"),
            EdgeMatrixPriorFactory.build(c, "edge_matrix"),
        ),
        "uniform_hyperprior": lambda c: UnavailableOption(c.degrees.name),
    }


if __name__ == "__main__":
    pass
