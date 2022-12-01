from __future__ import annotations
import typing

import midynet.metrics
from pyhectiqlab import Config
from .factory import Factory, OptionError, MissingRequirementsError

__all__ = ("MetricsConfig", "MetricsCollectionConfig", "MetricsFactory")


class MetricsConfig(Config):
    @classmethod
    def monte_carlo(cls, name: str):
        return cls(name, num_samples=100, error_type="confidence")

    @classmethod
    def mcmc(cls, name: str):
        obj = cls(
            name,
            num_sweeps=1000,
            error_type="percentile",
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

    # @classmethod
    # def partition_mcmc(cls, name: str):
    #     return cls(
    #         name,
    #         num_sweeps=1000,
    #         burn_per_vertex=5,
    #         equilibriate=True,
    #         start_from_original=False,
    #         num_iter=10,
    #         num_betas=10,
    #         exp_betas=0.5,
    #     )

    # @classmethod
    # def data_entropy(cls):
    #     return cls.mcmc("data_entropy")
    #
    # @classmethod
    # def data_prediction_entropy(cls):
    #     return cls.mcmc("data_prediction_entropy")
    #
    # @classmethod
    # def graph_entropy(cls):
    #     return cls.mcmc("graph_entropy")
    #
    # @classmethod
    # def graph_reconstruction_entropy(cls):
    #     return cls.mcmc("graph_reconstruction_entropy")
    #
    # @classmethod
    # def reconstructability(cls):
    #     return cls.mcmc("reconstructability")
    #
    # @classmethod
    # def predictability(cls):
    #     return cls.mcmc("predictability")

    @classmethod
    def recon_information(cls):
        return cls.mcmc("recon_information")

    @classmethod
    def heuristics(cls):
        return cls(
            "heuristics",
            method="correlation",
            num_samples=100,
            error_type="percentile",
        )


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
            "metrics_names",
            config_types,
            force_non_sequence=True,
            unique=True,
        )
        return obj


class MetricsFactory(Factory):
    @classmethod
    def build(cls, config: Config) -> midynet.metrics.Metrics:
        if issubclass(type(config), Config) and config.unmet_requirements():
            raise MissingRequirementsError(config)
        options = {
            k[6:]: getattr(cls, k)
            for k in cls.__dict__.keys()
            if k[:6] == "build_"
        }
        metrics = config.metrics
        if isinstance(metrics, MetricsConfig):
            if metrics.name in options:
                return options[metrics.name](config)
            else:
                raise OptionError(
                    actual=metrics.name, expected=list(options.keys())
                )
        elif isinstance(metrics, MetricsCollectionConfig):
            collections = {}
            for name in metrics.metrics_names:
                if name in options:
                    collections[name] = options[name](config)
                else:
                    raise OptionError(
                        actual=name, expected=list(options.keys())
                    )
            return collections
        else:
            message = (
                f"Invalid type {type(config)} for building metrics,"
                + "must contain type"
                + "`[MetricsConfig, MetricsCollectionConfig]`."
            )
            raise TypeError(message)

    # @staticmethod
    # def build_data_entropy(config: MetricsCollectionConfig):
    #     return midynet.metrics.DataEntropyMetrics(config)
    #
    # @staticmethod
    # def build_data_prediction_entropy(config: MetricsCollectionConfig):
    #     return midynet.metrics.DataPredictionEntropyMetrics(config)
    #
    # @staticmethod
    # def build_predictability(config: MetricsCollectionConfig):
    #     return midynet.metrics.PredictabilityMetrics(config)
    #
    # @staticmethod
    # def build_graph_entropy(config: MetricsCollectionConfig):
    #     return midynet.metrics.GraphEntropyMetrics(config)
    #
    # @staticmethod
    # def build_graph_reconstruction_entropy(config: MetricsCollectionConfig):
    #     return midynet.metrics.GraphReconstructionEntropyMetrics(config)
    #
    # @staticmethod
    # def build_reconstructability(config: MetricsCollectionConfig):
    #     return midynet.metrics.ReconstructabilityMetrics(config)

    @staticmethod
    def build_recon_information(config: MetricsCollectionConfig):
        return midynet.metrics.ReconstructionInformationMeasuresMetrics(
            config
        )

    @staticmethod
    def build_heuristics(config: MetricsCollectionConfig):
        return midynet.metrics.ReconstructionHeuristicsMetrics(config)


if __name__ == "__main__":
    pass
