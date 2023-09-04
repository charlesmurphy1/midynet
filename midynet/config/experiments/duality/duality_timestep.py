import os
import numpy as np
import dotenv

from midynet.config import GraphConfig, DataModelConfig, ExperimentConfig

dotenv.load_dotenv(os.path.join(os.path.expanduser("~"), ".md-env"))


class DualityTimestepScriptConfig(ExperimentConfig):
    T = np.unique(np.logspace(2, 4, 100).astype("int")).tolist()
    data_model_dict = {
        "sis": DataModelConfig.sis(
            infection_prob=[0.25, 0.5, 1.0],
            recovery_prob=0.1,
            auto_activation_prob=1e-4,
        ),
        "glauber": DataModelConfig.glauber(coupling=[0.25, 0.5, 1.0]),
        "cowan": DataModelConfig.cowan(nu=[0.5, 1.0, 2.0], eta=0.1),
    }

    @staticmethod
    def default(
        save_path,
        data_model="glauber",
        n_workers=1,
        n_samples_per_worker=100,
        n_async_jobs=1,
        seed=None,
    ):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        prior = GraphConfig.erdosrenyi(
            size=5,
            edge_count=5,
            loopy=False,
            multigraph=False,
        )
        config = ExperimentConfig.default(
            f"duality-timestep-{data_model}",
            data_model=DualityTimestepScriptConfig.data_model_dict[
                data_model
            ],
            prior=prior,
            metrics=[
                "pastinfo",
            ],
            path=save_path,
            n_workers=n_workers,
            n_async_jobs=n_async_jobs,
            seed=seed,
        )
        config.data_model.length = DualityTimestepScriptConfig.T
        config.metrics.pastinfo.past_length = [0.0, 0.5, -5.0]
        config.metrics.pastinfo.n_samples = n_samples_per_worker * n_workers
        config.metrics.pastinfo.data_mcmc.method = "exact"
        config.lock()
        return config

    @staticmethod
    def all(name="duality-timesteps", **kwargs):
        path = lambda m: os.path.join(
            os.getenv("MD-DATA_PATH", "./"), name, m
        )
        return [
            DualityTimestepScriptConfig.default(
                save_path=path(m),
                data_model=m,
                **kwargs,
            )
            for m in DualityTimestepScriptConfig.data_model_dict.keys()
        ]

    @staticmethod
    def test(**kwargs):
        DualityTimestepScriptConfig.T = DualityTimestepScriptConfig.T[::10]
        return DualityTimestepScriptConfig.all(
            name="test-duality-timesteps",
            n_workers=kwargs.pop("n_workers", 1),
            n_samples_per_worker=kwargs.pop("n_samples_per_worker", 1),
            **kwargs,
        )
