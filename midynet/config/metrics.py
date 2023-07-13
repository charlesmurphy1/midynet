from __future__ import annotations

from typing import Union

import midynet.metrics
from midynet.config import Config, static

from .factory import Factory, MissingRequirementsError, OptionError

__all__ = ("MetricsConfig", "MetricsCollectionConfig", "MetricsFactory")


@static
class MCMCDataConfig(Config):
    @classmethod
    def exact(cls, **kwargs):
        return cls("data_mcmc", method="exact", reset_original=True, **kwargs)

    @classmethod
    def meanfield(cls, **kwargs):
        return cls(
            "data_mcmc",
            method="meanfield",
            n_sweeps=1000,
            n_gibbs_sweeps=10,
            n_steps_per_vertex=1,
            burn_sweeps=4,
            sample_prior=True,
            sample_params=False,
            start_from_original=False,
            reset_original=True,
            **kwargs,
        )

    @classmethod
    def annealed(cls, **kwargs):
        return cls(
            "annealed",
            method="annealed",
            n_sweeps=1000,
            n_gibbs_sweeps=10,
            n_steps_per_vertex=1,
            burn_sweeps=4,
            sample_prior=True,
            sample_params=False,
            start_from_original=False,
            reset_original=True,
            n_betas=10,
            exp_betas=0.5,
            **kwargs,
        )


@static
class MCMCGraphConfig(Config):
    @classmethod
    def exact(cls, **kwargs):
        return cls("exact", method="exact", reset_original=True, **kwargs)

    @classmethod
    def meanfield(cls, **kwargs):
        return cls(
            "meanfield",
            method="partition_meanfield",
            n_sweeps=1000,
            n_steps_per_vertex=5,
            n_gibbs_sweeps=10,
            burn_sweeps=5,
            start_from_original=False,
            reset_original=True,
            equilibriate_mode_cluster=False,
            **kwargs,
        )


@static
class MetricsConfig(Config):
    @classmethod
    def mcmc(
        cls,
        name: str,
        graph_mcmc: MCMCGraphConfig or str = "meanfield",
        data_mcmc: MCMCDataConfig or str = "meanfield",
        reduction="normal",
        n_samples=100,
        **kwargs,
    ):
        graph_mcmc = (
            getattr(MCMCGraphConfig, graph_mcmc)()
            if isinstance(graph_mcmc, str)
            else graph_mcmc
        )
        data_mcmc = (
            getattr(MCMCDataConfig, data_mcmc)()
            if isinstance(data_mcmc, str)
            else data_mcmc
        )
        return cls(
            name,
            graph_mcmc=graph_mcmc,
            data_mcmc=data_mcmc,
            reduction=reduction,
            n_samples=n_samples,
            **kwargs,
        )

    @classmethod
    def reconinfo(cls, **kwargs):
        return cls.mcmc("reconinfo", **kwargs)

    @classmethod
    def efficiency(cls, **kwargs):
        config = cls.mcmc("efficiency", resample_graph=False, **kwargs)
        return config

    @classmethod
    def reconheuristics(cls):
        return cls(
            "reconheuristics",
            method="correlation",
            n_samples=100,
            reduction="normal",
        )

    @classmethod
    def susceptibility(cls):
        return cls("susceptibility", n_samples=100, reduction="identity")

    @classmethod
    def recon_error(cls, **kwargs):
        return cls.mcmc(
            "recon_error",
            reconstructor=kwargs.pop("reconstructor", "bayesian"),
            measures=kwargs.pop(
                "measures", "roc, posterior_similarity, accuracy"
            ),
            **kwargs,
        )

    @classmethod
    def pred_error(cls, **kwargs):
        return cls(
            "pred_error",
            predictor=kwargs.pop("predictor", "average_probability"),
            n_samples=kwargs.pop("n_samples", 100),
            measures=kwargs.pop("measures", "absolute_error"),
            **kwargs,
        )


@static
class MetricsCollectionConfig(Config):
    @classmethod
    def auto(cls, configs: Union[str, list[str], list[MetricsConfig]]):
        if not isinstance(configs, list):
            configs = [configs]
        configs = [
            getattr(MetricsConfig, c)() if isinstance(c, str) else c
            for c in configs
        ]
        configs = {c.name: c for c in configs}

        config = cls(
            "metrics",
            **configs,
        )
        config._state["metrics_names"] = list(configs.keys())
        config.__types__["metrics_names"] = str
        config.not_sequence("metrics_names")
        return config


class MetricsFactory(Factory):
    @classmethod
    def build(cls, config: Config) -> midynet.metrics.Metrics:
        options = {
            k[6:]: getattr(cls, k)
            for k in cls.__dict__.keys()
            if k[:6] == "build_"
        }
        metrics = config.metrics
        if "metrics_names" in metrics:
            collections = {}
            for name in metrics.metrics_names:
                if name in options:
                    collections[name] = options[name]()
                else:
                    raise OptionError(
                        actual=name, expected=list(options.keys())
                    )
            return collections
        else:
            if metrics.name in options:
                return options[metrics.name]()
            else:
                raise OptionError(
                    actual=metrics.name, expected=list(options.keys())
                )

    @staticmethod
    def build_reconinfo():
        return midynet.metrics.ReconstructionInformationMeasuresMetrics()

    @staticmethod
    def build_efficiency():
        return midynet.metrics.ReconstructionEfficiencyMetrics()

    @staticmethod
    def build_susceptibility():
        return midynet.metrics.SusceptibilityMetrics()

    @staticmethod
    def build_recon_error():
        return midynet.metrics.ReconstructionErrorMetrics()

    @staticmethod
    def build_pred_error():
        return midynet.metrics.PredictionErrorMetrics()


if __name__ == "__main__":
    pass
