from __future__ import annotations
import typing

import midynet.metrics
from midynet.config import Config, static
from .factory import Factory, OptionError, MissingRequirementsError

__all__ = ("MetricsConfig", "MetricsCollectionConfig", "MetricsFactory")


@static
class MetricsConfig(Config):
    @classmethod
    def mcmc(cls, name: str):
        obj = cls(
            name,
            num_sweeps=1000,
            sweep_type="metropolis",
            graph_rate=1,
            prior_rate=0,
            param_rate=0,
            reduction="normal",
            method="meanfield",
            num_samples=100,
            burn_per_vertex=5,
            initial_burn=2000,
            start_from_original=False,
            K=10,
            num_betas=10,
            exp_betas=0.5,
            equilibrate_mode_cluster=False,
        )
        return obj

    @classmethod
    def reconinfo(cls):
        return cls.mcmc("reconinfo")

    @classmethod
    def targreconinfo(cls):
        return cls.mcmc("targreconinfo")

    @classmethod
    def reconheuristics(cls):
        return cls(
            "reconheuristics",
            method="correlation",
            num_samples=100,
            reduction="normal",
        )

    @classmethod
    def linregheur(cls):
        return cls(
            "linregheur",
            graph_features="all",
            state_features="mean",
            num_samples=100,
            reduction="normal",
        )

    @classmethod
    def miheur(cls):
        return cls(
            "miheur",
            graph_features="all",
            state_features="mean",
            num_samples=100,
            reduction="normal",
        )

    @classmethod
    def susceptibility(cls):
        return cls("susceptibility", num_samples=100, reduction="identity")


@static
class MetricsCollectionConfig(Config):
    @classmethod
    def auto(cls, config_types: typing.Union[str, list[str]]):
        if isinstance(config_types, str):
            config_types = [config_types]
        config = cls(
            "metrics",
            **{a: getattr(MetricsConfig, a)() for a in config_types},
        )
        config._state["metrics_names"] = config_types
        config.__types__["metrics_names"] = str
        config.not_sequence("metrics_names")
        return config


class MetricsFactory(Factory):
    @classmethod
    def build(cls, config: Config) -> midynet.metrics.Metrics:
        options = {
            k[6:]: getattr(cls, k) for k in cls.__dict__.keys() if k[:6] == "build_"
        }
        metrics = config.metrics
        if "metrics_names" in metrics:
            collections = {}
            for name in metrics.metrics_names:
                if name in options:
                    collections[name] = options[name]()
                else:
                    raise OptionError(actual=name, expected=list(options.keys()))
            return collections
        else:
            if metrics.name in options:
                return options[metrics.name]()
            else:
                raise OptionError(actual=metrics.name, expected=list(options.keys()))

    @staticmethod
    def build_reconinfo():
        return midynet.metrics.ReconstructionInformationMeasuresMetrics()

    @staticmethod
    def build_targreconinfo():
        return midynet.metrics.TargetedReconstructionInformationMeasuresMetrics()

    @staticmethod
    def build_reconheur():
        return midynet.metrics.ReconstructionHeuristicsMetrics()

    @staticmethod
    def build_linregheur():
        return midynet.metrics.LinearRegressionHeuristicsMetrics()

    @staticmethod
    def build_miheur():
        return midynet.metrics.MutualInformationHeuristicsMetrics()

    @staticmethod
    def build_susceptibility():
        return midynet.metrics.SusceptibilityMetrics()


if __name__ == "__main__":
    pass
