from __future__ import annotations
import pathlib
import tempfile

from typing import List, Optional, Union

from midynet.config import (
    Config,
    static,
    DataModelConfig,
    GraphConfig,
    MetricsCollectionConfig,
)

__all__ = ("ExperimentConfig",)


class ExperimentConfig(Config):
    @classmethod
    def test(cls) -> ExperimentConfig:
        config = cls(name="test")
        config.data_model = DataModelConfig.auto(
            "sis", length=5, infection_prob=[0, 0.5, 1.0]
        )

        config.prior = GraphConfig.auto("erdosrenyi", size=5, edge_count=5)
        config.metrics = MetricsCollectionConfig.auto("recon_information")
        config.metrics.recon_information.n_samples = 1
        config.metrics.recon_information.n_sweeps = 10
        config.metrics.recon_information.initial_burn = 10
        config.path = tempfile.mktemp()
        config.n_workers = 1
        config.seed = 1

        config.lock_types()

        return config

    @classmethod
    def default(
        cls,
        name: str,
        data_model: str,
        prior: str,
        target: str = "None",
        metrics: Optional[List[str]] = None,
        path: Union[str, pathlib.Path] = ".",
        n_workers: int = 1,
        n_async_jobs: int = 1,
        seed: Optional[int] = None,
        data_model_params=None,
        graph_params=None,
        target_params=None,
    ) -> ExperimentConfig:
        data_model_params = {} if data_model_params is None else data_model_params
        graph_params = {} if graph_params is None else graph_params
        target_params = {} if target_params is None else target_params
        config = cls(name=name)
        config.data_model = DataModelConfig.auto(data_model, **data_model_params)
        config.target = (
            GraphConfig.auto(target, **target_params) if target != "None" else "None"
        )
        config.prior = GraphConfig.auto(prior, **graph_params)
        if config.target != "None":
            config.prior.from_target(config.target)
        config.metrics = MetricsCollectionConfig.auto(
            metrics if metrics is not None else []
        )
        config.path = str(path)
        config.n_workers = n_workers
        config.n_async_jobs = n_async_jobs
        config.seed = seed
        config.resources = Config(name="resources")
        config.lock_types()

        return config


if __name__ == "__main__":
    pass
