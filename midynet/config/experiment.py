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
        config.metrics.recon_information.num_samples = 1
        config.metrics.recon_information.num_sweeps = 10
        config.metrics.recon_information.initial_burn = 10
        config.path = tempfile.mktemp()
        config.num_procs = 1
        config.seed = 1

        config.lock_types()

        return config

    @classmethod
    def reconstruction(
        cls,
        name: str,
        data_model: str,
        prior: str,
        target: str = "None",
        metrics: Optional[List[str]] = None,
        path: Union[str, pathlib.Path] = ".",
        num_procs: int = 1,
        seed: Optional[int] = None,
        data_model_params=None,
        graph_params=None,
        target_params=None,
    ) -> ExperimentConfig:
        data_model_params = (
            {} if data_model_params is None else data_model_params
        )
        graph_params = {} if graph_params is None else graph_params
        target_params = {} if target_params is None else target_params
        config = cls(name=name)
        config.data_model = DataModelConfig.auto(
            data_model, **data_model_params
        )
        config.prior = GraphConfig.auto(prior, **graph_params)
        config.target = (
            GraphConfig.auto(target, **target_params)
            if target != "None"
            else "None"
        )
        config.metrics = MetricsCollectionConfig.auto(
            metrics if metrics is not None else []
        )
        for m in config.metrics.metrics_names:
            ns = config.metrics.get(m).num_samples
            config.metrics.get(m).num_samples = (
                max(1, ns // num_procs) * num_procs
            )
        config.path = str(path)
        config.num_procs = num_procs
        config.seed = seed
        config.resources = dict()
        config.lock_types()

        return config


if __name__ == "__main__":
    pass
