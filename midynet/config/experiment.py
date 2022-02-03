import pathlib
import time

from midynet.config import (
    Config,
    DynamicsConfig,
    RandomGraphConfig,
    MetricsCollectionConfig,
)

__all__ = ("ExperimentConfig",)


class ExperimentConfig(Config):
    requirements: set[str] = {
        "name",
        "dynamics",
        "graph",
        "metrics",
        "path",
        "num_procs",
        "seed",
    }

    @classmethod
    def default(
        cls,
        name: str,
        dynamics: str,
        graph: str,
        metrics: list[str] = None,
        path: pathlib.Path = ".",
        num_procs: int = 1,
        seed: int = None,
    ):
        obj = cls(name=name)
        obj.insert("dynamics", DynamicsConfig.auto(dynamics))
        obj.insert("graph", RandomGraphConfig.auto(graph))
        obj.insert(
            "metrics",
            MetricsCollectionConfig.auto(
                metrics if metrics is not None else []
            ),
        )
        for m in obj.metrics.metrics_names:
            ns = obj.metrics.get_value(m).num_samples
            obj.metrics.get_value(m).set_value(
                "num_samples", max(1, ns // num_procs) * num_procs
            )
        obj.insert(
            "path",
            path if isinstance(path, pathlib.Path) else pathlib.Path(path),
            force_non_sequence=True,
            unique=True,
        )
        obj.insert(
            "num_procs", num_procs, force_non_sequence=True, unique=True
        )
        obj.insert(
            "seed",
            seed or int(time.time()),
            force_non_sequence=True,
            unique=True,
        )

        return obj


if __name__ == "__main__":
    pass
