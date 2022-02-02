from __future__ import annotations
import typing
import midynet.metrics
from typing import Union
from .config import Config
from .factory import Factory, OptionError, MissingRequirementsError

__all__ = ("MetricsConfig", "MetricsCollectionConfig", "MetricsFactory")


class MetricsConfig(Config):
    @classmethod
    def monte_carlo(cls):
        return cls(num_samples=100, error_type="confidence")

    @classmethod
    def mcmc(cls):
        obj = cls(
            num_sweeps=1000,
            error_type="confidence",
            method="meanfield",
            num_samples=100,
            burn_per_vertex=5,
            initial_burn=2000,
            reset_to_original=True,
            K=10,
            num_betas=10,
            exp_betas=0.5,
        )
        return obj

    @classmethod
    def dynamics_entropy(cls):
        obj = cls.mcmc()
        obj.set_value("name", "dynamics_entropy")
        return obj

    @classmethod
    def dynamics_prediction_entropy(cls):
        obj = cls.monte_carlo()
        obj.set_value("name", "dynamics_prediction_entropy")
        return obj

    @classmethod
    def graph_entropy(cls):
        obj = cls.monte_carlo()
        obj.set_value("name", "graph_entropy")
        return obj

    @classmethod
    def graph_reconstruction_entropy(cls):
        obj = cls.mcmc()
        obj.set_value("name", "graph_reconstruction_entropy")
        return obj

    @classmethod
    def reconstructability(cls):
        obj = cls.mcmc()
        obj.set_value("name", "reconstructability")
        return obj

    @classmethod
    def predictability(cls):
        obj = cls.mcmc()
        obj.set_value("name", "predictability")
        return obj

    @classmethod
    def mutualinfo(cls):
        obj = cls.mcmc()
        obj.set_value("name", "mutualinfo")
        return obj


class MetricsCollectionConfig(Config):
    unique_parameters: list[str] = {
        "name",
        "metrics_names",
        "num_procs",
        "seed",
    }

    @classmethod
    def auto(cls, config_types: typing.Union[str, list[str]]):
        if isinstance(config_types, str):
            config_types = [config_types]
        obj = cls(**{a: MetricsConfig.auto(a) for a in config_types})
        obj.insert(
            "metrics_names", config_types, force_non_sequence=True, unique=True
        )
        return obj


class MetricsFactory(Factory):
    @classmethod
    def build(
        cls, config: Union[MetricsConfig, MetricsCollectionConfig]
    ) -> midynet.metrics.Metrics:
        if issubclass(type(config), Config) and config.unmet_requirements():
            raise MissingRequirementsError(config)
        options = {
            k[6:]: getattr(cls, k)
            for k in cls.__dict__.keys()
            if k[:6] == "build_"
        }
        if isinstance(config, MetricsConfig):
            if config.name in options:
                return options[config.name](config)
            else:
                raise OptionError(
                    actual=config.name, expected=list(options.keys())
                )
        elif isinstance(config, MetricsCollectionConfig):
            metrics = {}
            for name in config.metrics.metrics_names:
                if name in options:
                    metrics[name] = options[name](config)
                else:
                    raise OptionError(
                        actual=name, expected=list(options.keys())
                    )
            return metrics
        else:
            message = (
                f"Invalid type {type(config)} for building metrics,"
                + "expected types `[MetricsConfig, MetricsCollectionConfig]`."
            )
            raise TypeError(message)

    @staticmethod
    def build_dynamics_entropy(config: MetricsCollectionConfig):
        return midynet.metrics.DynamicsEntropyMetrics(config)

    @staticmethod
    def build_dynamics_prediction_entropy(config: MetricsCollectionConfig):
        return midynet.metrics.DynamicsPredictionEntropyMetrics(config)

    @staticmethod
    def build_predictability(config: MetricsCollectionConfig):
        return midynet.metrics.PredictabilityMetrics(config)

    @staticmethod
    def build_graph_entropy(config: MetricsCollectionConfig):
        return midynet.metrics.GraphEntropyMetrics(config)

    @staticmethod
    def build_graph_reconstruction_entropy(config: MetricsCollectionConfig):
        return midynet.metrics.GraphReconstructionEntropyMetrics(config)

    @staticmethod
    def build_reconstructability(config: MetricsCollectionConfig):
        return midynet.metrics.ReconstructabilityMetrics(config)

    @staticmethod
    def build_mutualinfo(config: MetricsCollectionConfig):
        return midynet.metrics.MutualInformationMetrics(config)


if __name__ == "__main__":
    pass
