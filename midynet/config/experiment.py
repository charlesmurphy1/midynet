import pathlib
import time

from midynet.config import *


__all__ = ["ExperimentConfig"]


class ExperimentConfig(Config):
    requirements: set[str] = {
        "name",
        "dynamics",
        "graph",
        "metrics",
        "path",
        "seed",
        "num_procs",
    }

    @classmethod
    def default(
        cls,
        name: str,
        dynamics: str,
        graph: str,
        metrics: list[str] = None,
        path: pathlib.Path = ".",
        seed: int = None,
        num_procs: int = 1,
    ):
        obj = cls(name=name)
        obj.insert("dynamics", DynamicsConfig.auto(dynamics))
        obj.insert("graph", RandomGraphConfig.auto(graph))
        graph_name = obj.graph.name.split("_")[-1]
        obj.insert("mcmc", MCMCConfig.auto(graph_name, obj.graph))
        obj.insert(
            "metrics",
            MetricsCollectionConfig.auto(metrics if metrics is not None else []),
        )
        obj.insert(
            "path", path if isinstance(path, pathlib.Path) else pathlib.Path(path)
        )
        obj.insert("seed", seed if seed is not None else int(time.time()))
        obj.insert("num_procs", seed if seed is not None else int(time.time()))

        return obj


if __name__ == "__main__":
    pass
