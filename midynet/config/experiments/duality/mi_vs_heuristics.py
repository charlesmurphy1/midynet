import os
import dotenv

from midynet.config import (
    ExperimentConfig,
    DataModelConfig,
    GraphConfig,
)
from midynet.config.experiments.util import format_sequence

dotenv.load_dotenv(os.path.join(os.path.expanduser("~"), ".md-env"))


class PredHeuristicsScriptConfig(ExperimentConfig):
    coupling = format_sequence((0, 0.5, 20), (0.5, 2.0, 10))

    @staticmethod
    def default(
        save_path,
        n_workers=1,
        n_async_jobs=1,
        seed=None,
    ):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model = DataModelConfig.glauber(
            length=100, coupling=PredHeuristicsScriptConfig.coupling
        )
        prior = GraphConfig.erdosrenyi(
            size=100, edge_count=250, loopy=False, multigraph=False
        )
        config = ExperimentConfig.default(
            f"mi_vs_heuristics",
            model,
            prior,
            metrics=[
                "pred_error",
            ],
            path=save_path,
            n_workers=n_workers,
            n_async_jobs=n_async_jobs,
            seed=seed,
        )
        config.metrics.pred_error.predictor = [
            "logistic",
            "mlp",
        ]
        config.metrics.pred_error.n_samples = n_workers // n_async_jobs
        config.lock()
        return config

    @staticmethod
    def all(name="mi_vs_heuristics", **kwargs):
        path = os.path.join(os.getenv("MD-DATA_PATH", "./"), name)
        return [
            PredHeuristicsScriptConfig.default(
                save_path=path,
                **kwargs,
            )
        ]

    @staticmethod
    def test(**kwargs):
        PredHeuristicsScriptConfig.coupling = (
            PredHeuristicsScriptConfig.coupling[::10]
        )
        return PredHeuristicsScriptConfig.all(
            name="test-mi_vs_heuristics",
            n_workers=kwargs.pop("n_workers", 1),
            **kwargs,
        )


class ReconHeuristicsScriptConfig(ExperimentConfig):
    coupling = format_sequence((0, 0.5, 20), (0.5, 2.0, 10))

    @staticmethod
    def default(
        save_path,
        n_workers=1,
        n_async_jobs=1,
        seed=None,
    ):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model = DataModelConfig.glauber(
            length=100, coupling=ReconHeuristicsScriptConfig.coupling
        )
        prior = GraphConfig.erdosrenyi(
            size=100, edge_count=250, loopy=False, multigraph=False
        )
        config = ExperimentConfig.default(
            f"mi_vs_heuristics",
            model,
            prior,
            metrics=[
                "recon_error",
            ],
            path=save_path,
            n_workers=n_workers,
            n_async_jobs=n_async_jobs,
            seed=seed,
        )
        config.metrics.recon_error.reconstructor = [
            "transfer_entropy",
            "granger_causality",
            "correlation",
        ]
        config.metrics.recon_error.n_samples = n_workers // n_async_jobs
        config.lock()
        return config

    @staticmethod
    def all(name="mi_vs_heuristics", **kwargs):
        path = os.path.join(os.getenv("MD-DATA_PATH", "./"), name)
        return [
            ReconHeuristicsScriptConfig.default(
                save_path=path,
                **kwargs,
            )
        ]

    @staticmethod
    def test(**kwargs):
        ReconHeuristicsScriptConfig.coupling = (
            ReconHeuristicsScriptConfig.coupling[::10]
        )
        return ReconHeuristicsScriptConfig.all(
            name="test-mi_vs_heuristics",
            n_workers=kwargs.pop("n_workers", 1),
            **kwargs,
        )
