from __future__ import annotations
from typing import Set, Any
from midynet.data.dynamics import (
    Dynamics,
    CowanDynamics,
    DegreeDynamics,
    SISDynamics,
    GlauberDynamics,
)
from .config import Config
from .factory import Factory, OptionError

__all__ = ("DataModelConfig", "DataModelFactory")


class DataModelConfig(Config):
    @classmethod
    def glauber(
        cls,
        num_steps: int = 100,
        coupling: float = 1.0,
        auto_activation_prob=0,
        auto_deactivation_prob=0,
        normalize: bool = True,
        num_active: int = -1,
    ) -> DynamicsConfig:
        return cls(
            name="glauber",
            num_steps=num_steps,
            coupling=coupling,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
            normalize=normalize,
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
        num_steps: int = 100,
        infection_prob: float = 0.1,
        recovery_prob: float = 0.1,
        auto_activation_prob=0.001,
        auto_deactivation_prob=0,
        normalize: bool = True,
        num_active: int = 1,
    ) -> DynamicsConfig:
        return cls(
            name="sis",
            num_steps=num_steps,
            infection_prob=infection_prob,
            recovery_prob=recovery_prob,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
            normalize=normalize,
            num_active=num_active,
        )

    @classmethod
    def cowan(
        cls,
        num_steps: int = 100,
        nu: float = 1.0,
        a: float = 8.0,
        mu: float = 1.0,
        eta: float = 0.5,
        auto_activation_prob=0,
        auto_deactivation_prob=0,
        normalize: bool = True,
        num_active: int = 1,
    ) -> DynamicsConfig:
        return cls(
            name="cowan",
            num_steps=num_steps,
            nu=nu,
            a=a,
            mu=mu,
            eta=eta,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
            normalize=normalize,
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
        num_steps: int = 100,
        C: float = 1.0,
        auto_activation_prob=0,
        auto_deactivation_prob=0,
        num_active: int = 2**31 - 1,
    ) -> DynamicsConfig:
        return cls(
            name="degree",
            num_steps=num_steps,
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
            num_steps=config.num_steps,
            coupling=config.coupling,
            auto_activation_prob=config.auto_activation_prob,
            auto_deactivation_prob=config.auto_deactivation_prob,
            async_mode=False,
            normalize=config.normalize,
            num_active=config.num_active,
        )

    @staticmethod
    def build_sis(config: DataModelConfig):
        return SISDynamics(
            num_steps=config.num_steps,
            infection_prob=config.infection_prob,
            recovery_prob=config.recovery_prob,
            auto_activation_prob=config.auto_activation_prob,
            auto_deactivation_prob=config.auto_deactivation_prob,
            async_mode=False,
            normalize=config.normalize,
            num_active=config.num_active,
        )

    @staticmethod
    def build_cowan(config: DataModelConfig):
        return CowanDynamics(
            num_steps=config.num_steps,
            nu=config.nu,
            a=config.a,
            mu=config.mu,
            eta=config.eta,
            auto_activation_prob=config.auto_activation_prob,
            auto_deactivation_prob=config.auto_deactivation_prob,
            async_mode=False,
            normalize=config.normalize,
            num_active=config.num_active,
        )

    @staticmethod
    def build_degree(config: DataModelConfig):
        return DegreeDynamics(num_steps=config.num_steps, C=config.C)
