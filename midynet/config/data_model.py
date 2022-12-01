from __future__ import annotations
from typing import Set, Any
from graphinf.data_model.dynamics import (
    Dynamics,
    CowanDynamics,
    DegreeDynamics,
    SISDynamics,
    GlauberDynamics,
)
from pyhectiqlab import Config
from .factory import Factory, OptionError

__all__ = ("DataModelConfig", "DataModelFactory")


class DataModelConfig(Config):
    @classmethod
    def glauber(
        cls,
        length: int = 100,
        coupling: float = 1.0,
        past_length: int = 0,
        initial_burn: int = 0,
        auto_activation_prob=0,
        auto_deactivation_prob=0,
        num_active: int = -1,
    ) -> DynamicsConfig:
        return cls(
            name="glauber",
            length=length,
            coupling=coupling,
            past_length=past_length,
            initial_burn=initial_burn,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
            num_active=num_active,
        )

    @classmethod
    def ising(cls, **kwargs):
        cfg = cls.glauber(**kwargs)
        cfg.set_value("name", "ising")
        return cfg

    @classmethod
    def sis(
        cls,
        length: int = 100,
        infection_prob: float = 0.1,
        recovery_prob: float = 0.1,
        past_length: int = 0,
        initial_burn: int = 0,
        auto_activation_prob=0.001,
        auto_deactivation_prob=0,
        num_active: int = 1,
    ) -> DynamicsConfig:
        return cls(
            name="sis",
            length=length,
            infection_prob=infection_prob,
            recovery_prob=recovery_prob,
            past_length=past_length,
            initial_burn=initial_burn,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
            num_active=num_active,
        )

    @classmethod
    def cowan(
        cls,
        length: int = 100,
        nu: float = 1.0,
        a: float = 8.0,
        mu: float = 1.0,
        eta: float = 0.1,
        past_length: int = 0,
        initial_burn: int = 0,
        auto_activation_prob=0,
        auto_deactivation_prob=0,
        num_active: int = 1,
    ) -> DynamicsConfig:
        return cls(
            name="cowan",
            length=length,
            nu=nu,
            a=a,
            mu=mu,
            eta=eta,
            past_length=past_length,
            initial_burn=initial_burn,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
            num_active=num_active,
        )

    @classmethod
    def cowan_forward(cls, **kwargs):
        cfg = cls.cowan(**kwargs)
        cfg.set_value("num_active", 1)
        return cfg

    @classmethod
    def cowan_backward(cls, **kwargs):
        cfg = cls.cowan(**kwargs)
        cfg.set_value("num_active", -1)
        return cfg

    @classmethod
    def degree(
        cls,
        length: int = 100,
        C: float = 1.0,
        auto_activation_prob=0,
        auto_deactivation_prob=0,
        num_active: int = 2 ** 31 - 1,
    ) -> DynamicsConfig:
        return cls(
            name="degree",
            length=length,
            C=C,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
            num_active=num_active,
        )


class DataModelFactory(Factory):
    @classmethod
    def build(cls, config: Config) -> Any:
        if config.unmet_requirements():
            raise MissingRequirementsError(config)
        options = {
            k[6:]: getattr(cls, k) for k in cls.__dict__.keys() if k[:6] == "build_"
        }
        name = config.name
        if name in options:
            return options[name](config)
        else:
            raise OptionError(actual=name, expected=list(options.keys()))

    @staticmethod
    def build_glauber(config: DataModelConfig):
        return GlauberDynamics(
            length=config.length,
            coupling=config.coupling,
            auto_activation_prob=config.auto_activation_prob,
            auto_deactivation_prob=config.auto_deactivation_prob,
        )

    @staticmethod
    def build_sis(config: DataModelConfig):
        return SISDynamics(
            length=config.length,
            infection_prob=config.infection_prob,
            recovery_prob=config.recovery_prob,
            auto_activation_prob=config.auto_activation_prob,
            auto_deactivation_prob=config.auto_deactivation_prob,
        )

    @staticmethod
    def build_cowan(config: DataModelConfig):
        return CowanDynamics(
            length=config.length,
            nu=config.nu,
            a=config.a,
            mu=config.mu,
            eta=config.eta,
            auto_activation_prob=config.auto_activation_prob,
            auto_deactivation_prob=config.auto_deactivation_prob,
        )

    @staticmethod
    def build_degree(config: DataModelConfig):
        return DegreeDynamics(length=config.length, C=config.C)
