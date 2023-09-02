import os
import dotenv

from math import ceil
from midynet.config import ExperimentConfig, DataModelConfig, GraphConfig
from midynet.config.experiments.util import format_sequence

dotenv.load_dotenv()


class ErrorHeuristicsScriptConfig(ExperimentConfig):
    @classmethod
    def default(
        cls,
        save_path,
        n_workers=os.getenv("MD-N_WORKERS", 1),
        n_samples_per_worker=1,
        n_async_jobs=1,
        n_sweeps=1000,
        seed=None,
    ):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        prior = GraphConfig.erdosrenyi(size=100, edge_count=250)
        model = DataModelConfig.glauber(
            length=100, coupling=format_sequence((0, 0.8, 25))
        )
        config = ExperimentConfig.default(
            f"error-heuristics",
            data_model=model,
            prior=prior,
            target=prior,
            metrics=["recon_error"],
            path=save_path,
            n_workers=n_workers,
            n_async_jobs=n_async_jobs,
            seed=seed,
        )

        config.metrics.recon_error.n_samples = n_samples_per_worker * ceil(
            n_workers / n_async_jobs
        )
        config.metrics.recon_error.reconstructor = [
            "bayesian",
            # "peixoto",
            "correlation",
            "granger_causality",
            "transfer_entropy",
        ]
        config.metrics.recon_error.data_mcmc.n_sweeps = n_sweeps
        config.metrics.recon_error.graph_mcmc.n_sweeps = n_sweeps
        config.metrics.recon_error.measures = (
            "roc, posterior_similarity, accuracy, mean_error"
        )
        config.metrics.recon_error.reduction = "identity"
        config.lock()
        return config

    @staticmethod
    def all(
        name="error-heuristics",
        **kwargs,
    ):
        exp = ErrorHeuristicsScriptConfig.default(
            save_path=os.path.join(os.getenv("MD-DATA_PATH", "./"), name),
            **kwargs,
        )
        return [exp]

    @staticmethod
    def test(**kwargs):
        return ErrorHeuristicsScriptConfig.all(
            name="test-error-heuristics",
            n_sweeps=kwargs.pop("n_sweeps", 10),
            **kwargs,
        )
