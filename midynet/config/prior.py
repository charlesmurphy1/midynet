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
    def delta(cls, edge_count: int = 0):
        if edge_count < 0:
            raise ValueError(f"`edge_count` must be non-negative")
        return cls(name="delta", state=edge_count)

    @classmethod
    def poisson(cls, mean: float = 0):
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
    def poisson(cls, mean: float = 1):
        if mean < 0:
            raise ValueError(f"`mean` must be non-negative.")
        return cls(name="poisson", mean=mean)

    @classmethod
    def uniform(cls, min: int = 1, max: int = None):
        if min is not None and max is None:
            min, max = 1, min
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
        elif config == "hyperuniform":
            return cls.hyperuniform()
        else:
            return super().auto(config)

    @classmethod
    def delta(cls, blocks: list[int]):
        return cls(name="delta", state=blocks)

    @classmethod
    def uniform(cls, size: int = 1):
        return cls(name="uniform", block_count=BlockCountPriorConfig.uniform(size))

    @classmethod
    def hyperuniform(cls, size: int = 1):
        return cls(name="hyperuniform", block_count=BlockCountPriorConfig.uniform(size))


class EdgeMatrixPriorConfig(Config):
    @classmethod
    def uniform(cls, edge_count):
        return cls(name="uniform", edge_count=EdgeCountPriorConfig.auto(edge_count))


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
    options: Dict[str, Callable[[Config], prior.sbm.EdgeCountPrior]] = {
        "delta": lambda c: prior.sbm.EdgeCountDeltaPrior(c.get_value("state", 0)),
        "poisson": lambda c: prior.sbm.EdgeCountPoissonPrior(c.get_value("mean", 0)),
    }


class BlockCountPriorFactory(Factory):
    options: Dict[str, Callable[[Config], prior.sbm.BlockCountPrior]] = {
        "delta": lambda c: prior.sbm.BlockCountDeltaPrior(c.get_value("state")),
        "poisson": lambda c: prior.sbm.BlockCountPoissonPrior(c.get_value("mean")),
        "uniform": lambda c: prior.sbm.BlockCountUniformPrior(
            c.get_value("min", 1), c.get_value("max", 1)
        ),
    }


class BlockPriorFactory(Factory):
    options: Dict[str, Callable[[Config], prior.sbm.BlockPrior]] = {
        "uniform": lambda c: prior.sbm.BlockUniformPrior(),
        "hyperuniform": lambda c: prior.sbm.BlockUniformHyperPrior(),
    }


class EdgeMatrixPriorFactory(Factory):
    options: Dict[str, Callable[[Config], prior.sbm.EdgeMatrixPrior]] = {
        "uniform": lambda c: prior.sbm.EdgeMatrixUniformPrior(),
    }


class DegreePriorFactory(Factory):
    options: Dict[str, Callable[[Config], prior.sbm.DegreePrior]] = {
        "uniform": lambda c: prior.sbm.DegreeUniformPrior(),
        "hyperuniform": lambda c: UnavailableOption(c.degrees.name),
    }


if __name__ == "__main__":
    pass
