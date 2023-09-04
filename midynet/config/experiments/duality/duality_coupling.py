import os
import numpy as np
import dotenv

from math import ceil
from midynet.config.experiments.util import format_sequence
from midynet.config import (
    GraphConfig,
    DataModelConfig,
    ExperimentConfig,
    MetricsConfig,
)

dotenv.load_dotenv(os.path.join(os.path.expanduser("~"), ".md-env"))


class DualityCouplingWithRealGraphsScriptConfig(ExperimentConfig):
    target_dict = {
        "glauber": (
            "littlerock",
            os.path.join(
                os.getenv("MD-DATA_PATH", "."), "/graphs/littlerock.pkl"
            ),
        ),
        "sis": (
            "euairlines",
            os.path.join(
                os.getenv("MD-DATA_PATH", "."), "/graphs/euairlines.pkl"
            ),
        ),
        "cowan_forward": (
            "celegans",
            os.path.join(
                os.getenv("MD-DATA_PATH", "."), "/graphs/celegans.pkl"
            ),
        ),
        "cowan_backward": (
            "celegans",
            os.path.join(
                os.getenv("MD-DATA_PATH", "."), "/graphs/celegans.pkl"
            ),
        ),
    }

    model_dict = {
        "glauber": DataModelConfig.glauber(
            length=2000,
            coupling=format_sequence(
                (0, 0.02, 5), (0.02, 0.04, 25), (0.04, 0.1, 5)
            ),
        ),
        "sis": DataModelConfig.sis(
            length=2000,
            recovery_prob=0.5,
            infection_prob=format_sequence((0, 0.02, 20), (0.02, 0.5, 30)),
        ),
        "cowan_forward": DataModelConfig.cowan_forward(
            length=5000,
            nu=format_sequence((0, 0.07, 5), (0.07, 0.2, 30), (0.2, 0.3, 5)),
        ),
        "cowan_backward": DataModelConfig.cowan_backward(
            length=5000, nu=format_sequence((0, 0.1, 25), (0.1, 0.3, 15))
        ),
    }

    @staticmethod
    def default(
        save_path,
        model,
        n_workers=1,
        n_async_jobs=1,
        n_sweeps=1000,
        seed=None,
    ):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        metrics = [
            MetricsConfig.bayesian(
                graph_mcmc=None,
                data_mcmc="meanfield",
                n_samples=n_workers // n_async_jobs,
                resample_graph=True,
            )
        ]
        (
            target,
            target_path,
        ) = DualityCouplingWithRealGraphsScriptConfig.target_dict[model]
        assert os.path.exists(
            target_path
        ), f"path {target_path} does not exist."
        config = ExperimentConfig.default(
            f"{target}-{model}",
            DualityCouplingWithRealGraphsScriptConfig.model_dict[model],
            "degree_constrained_configuration",
            target=target,
            metrics=metrics,
            path=save_path,
            n_workers=n_workers,
            n_async_jobs=n_async_jobs,
            seed=seed,
            target_params=dict(path=target_path),
        )

        config.prior.size = config.target.size
        if "backward" in model:
            config.data_model.n_active = config.prior.size
        elif model == "glauber":
            config.data_model.n_active = -1
        else:
            config.data_model.n_active = ceil(0.01 * config.prior.size)
        config.metrics.bayesian.data_mcmc.n_sweeps = n_sweeps
        config.metrics.bayesian.data_mcmc.n_steps_per_vertex = 1
        config.metrics.bayesian.data_mcmc.n_gibbs_sweeps = 5
        config.metrics.bayesian.data_mcmc.sample_prior = False
        config.metrics.bayesian.data_mcmc.burn_sweeps = 5
        config.lock()
        return config

    @staticmethod
    def all(name="duality-coupling-with-rn", **kwargs):
        path = lambda m: os.path.join(
            os.getenv("MD-DATA_PATH", "./"), name, m
        )
        return [
            DualityCouplingWithRealGraphsScriptConfig.default(
                save_path=path(m),
                model=m,
                **kwargs,
            )
            for m in DualityCouplingWithRealGraphsScriptConfig.model_dict.keys()
        ]

    @staticmethod
    def test(**kwargs):
        return DualityCouplingWithRealGraphsScriptConfig.all(
            name="test-duality-coupling-with-rn",
            n_workers=kwargs.pop("n_workers", 1),
            n_sweeps=kwargs.pop("n_sweeps", 10),
            **kwargs,
        )


class DualityCouplingSyntheticGraphsScriptConfig(ExperimentConfig):
    model_dict = {
        "glauber": DataModelConfig.glauber(
            length=2000,
            coupling=format_sequence(
                (0, 0.5, 6), (0.5, 0.75, 22), (0.75, 1.5, 6)
            ),
        ),
        "sis": DataModelConfig.sis(
            length=2000,
            recovery_prob=0.5,
            infection_prob=format_sequence((0, 0.125, 3), (0.0125, 0.5, 30)),
        ),
        "cowan_forward": DataModelConfig.cowan_forward(
            length=2000,
            nu=format_sequence((2.0, 2.5, 23), (2.5, 4, 10)),
        ),
        "cowan_backward": DataModelConfig.cowan_backward(
            length=2000,
            nu=format_sequence((1.0, 1.25, 4), (1.25, 1.5, 20), (1.5, 3, 10)),
        ),
    }

    @staticmethod
    def default(
        save_path,
        model,
        n_workers=1,
        n_async_jobs=1,
        n_sweeps=1000,
        seed=None,
    ):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        metrics = [
            MetricsConfig.bayesian(
                graph_mcmc=None,
                data_mcmc="meanfield",
                n_samples=n_workers // n_async_jobs,
            )
        ]

        prior = GraphConfig.nbinom(
            size=1000, edge_count=2500, heterogeneity=1
        )
        config = ExperimentConfig.default(
            model,
            DualityCouplingSyntheticGraphsScriptConfig.model_dict[model],
            prior,
            metrics=metrics,
            path=os.path.join(save_path, model),
            n_workers=n_workers,
            n_async_jobs=n_async_jobs,
            seed=seed,
        )

        if "backward" in model:
            config.data_model.n_active = config.prior.size
        elif model == "glauber":
            config.data_model.n_active = -1
        else:
            config.data_model.n_active = ceil(0.01 * config.prior.size)

        config.metrics.bayesian.data_mcmc.n_sweeps = n_sweeps
        config.metrics.bayesian.data_mcmc.n_steps_per_vertex = 1
        config.metrics.bayesian.data_mcmc.n_gibbs_sweeps = 5
        config.metrics.bayesian.data_mcmc.sample_prior = False
        config.metrics.bayesian.data_mcmc.burn_sweeps = 5
        config.lock()
        return config

    @staticmethod
    def all(name="duality-coupling-with-syn", **kwargs):
        path = lambda m: os.path.join(
            os.getenv("MD-DATA_PATH", "./"), name, m
        )
        return [
            DualityCouplingSyntheticGraphsScriptConfig.default(
                save_path=path(m),
                model=m,
                **kwargs,
            )
            for m in DualityCouplingSyntheticGraphsScriptConfig.model_dict.keys()
        ]

    @staticmethod
    def test(**kwargs):
        return DualityCouplingSyntheticGraphsScriptConfig.all(
            name="test-duality-coupling-with-rn",
            n_workers=kwargs.pop("n_workers", 1),
            n_sweeps=kwargs.pop("n_sweeps", 10),
            **kwargs,
        )
