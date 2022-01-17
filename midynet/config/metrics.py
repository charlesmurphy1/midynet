import numpy as np
import typing

from .config import Config
from .factory import Factory, OptionError
from .wrapper import Wrapper

from midynet.metrics import *

__all__ = ["MetricsConfig", "MetricsCollectionConfig", "MetricsFactory"]


class MetricsConfig(Config):
    @classmethod
    def dynamics_entropy(cls):
        obj = cls(
            name="dynamics_entropy",
            num_sweeps=100,
            method="meanfield",
            num_samples=100,
            K=10,
        )
        obj.insert("beta_k", np.linspace(0, 1, 10).tolist(), force_non_sequence=True)
        return obj

    @classmethod
    def dynamics_prediction_entropy(cls):
        return cls(name="dynamics_prediction_entropy", num_samples=100)

    @classmethod
    def predictability(cls):
        obj = cls(
            name="predictability",
            num_sweeps=100,
            method="meanfield",
            num_samples=100,
            K=10,
        )
        obj.insert("beta_k", np.linspace(0, 1, 10).tolist(), force_non_sequence=True)
        return obj

    @classmethod
    def graph_entropy(cls):
        return cls(name="graph_entropy", num_samples=100)

    @classmethod
    def graph_reconstruction_entropy(cls):
        obj = cls(
            name="graph_reconstruction_entropy",
            num_sweeps=100,
            method="meanfield",
            num_samples=100,
            K=10,
        )
        obj.insert("beta_k", np.linspace(0, 1, 10).tolist(), force_non_sequence=True)
        return obj

    @classmethod
    def reconstructability(cls):
        obj = cls(
            name="reconstructability",
            num_sweeps=100,
            method="meanfield",
            num_samples=100,
            K=10,
        )
        obj.insert("beta_k", np.linspace(0, 1, 10).tolist(), force_non_sequence=True)
        return obj


class MetricsCollectionConfig(Config):
    unique_parameters: list[str] = {"name", "metrics_names", "num_procs", "seed"}

    @classmethod
    def auto(cls, config_types: typing.Union[str, list[str]]):
        if isinstance(config_types, str):
            config_types = [config_types]
        obj = cls(**{a: MetricsConfig.auto(a) for a in config_types})
        obj.insert("metrics_names", config_types, force_non_sequence=True, unique=True)
        return obj


class MetricsFactory(Factory):
    @classmethod
    def build(cls, config: Config) -> typing.Any:
        if config.unmet_requirements() and not isinstance(config, str):
            raise MissingRequirementsError(config)
        options = {
            k[6:]: getattr(cls, k) for k in cls.__dict__.keys() if k[:6] == "build_"
        }
        if isinstance(config.metrics, MetricsConfig):
            if config.metrics.name in options:
                return options[config.name](config)
            else:
                raise OptionError(actual=config.name, expected=list(options.keys()))
        elif isinstance(config.metrics, MetricsCollectionConfig):
            names = config.metrics.metrics_names
            metrics = {}
            for name in config.metrics.metrics_names:
                if name in options:
                    metrics[name] = options[name](config)
                else:
                    raise OptionError(actual=name, expected=list(options.keys()))
            return metrics
        else:
            message = (
                f"Invalid type {type(config)} for building metrics,"
                + "expected types `[MetricsConfig, MetricsCollectionConfig]`."
            )
            raise TypeError(message)

    @staticmethod
    def build_dynamics_entropy(config: MetricsCollectionConfig):
        return DynamicsEntropyMetrics(config)

    @staticmethod
    def build_dynamics_prediction_entropy(config: MetricsCollectionConfig):
        return DynamicsPredictionEntropyMetrics(config)

    @staticmethod
    def build_predictability(config: MetricsCollectionConfig):
        return PredictabilityMetrics(config)

    @staticmethod
    def build_graph_entropy(config: MetricsCollectionConfig):
        return GraphEntropyMetrics(config)

    @staticmethod
    def build_graph_reconstruction_entropy(config: MetricsCollectionConfig):
        return GraphReconstructionEntropyMetrics(config)

    @staticmethod
    def build_reconstructability(config: MetricsCollectionConfig):
        return ReconstructabilityMetrics(config)


if __name__ == "__main__":
    pass
