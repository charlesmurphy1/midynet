from __future__ import annotations
from typing import Set, Any
from _midynet.dynamics import (
    Dynamics,
    CowanDynamics,
    DegreeDynamics,
    SISDynamics,
    GlauberDynamics,
    BlockLabeledDynamics,
    BlockLabeledCowanDynamics,
    BlockLabeledDegreeDynamics,
    BlockLabeledSISDynamics,
    BlockLabeledGlauberDynamics,
    NestedBlockLabeledCowanDynamics,
    NestedBlockLabeledDegreeDynamics,
    NestedBlockLabeledSISDynamics,
    NestedBlockLabeledGlauberDynamics,
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
        num_active: int = 2**31 - 1,
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
        cfg.set_value("num_active", 2**31 - 1)
        return cfg

    @classmethod
    def degree(
        cls,
        num_steps: int = 100,
        C: float = 1.0,
        auto_activation_prob=0,
        auto_deactivation_prob=0,
        normalize: bool = True,
        num_active: int = 2**31 - 1,
    ) -> DynamicsConfig:
        return cls(
            name="degree",
            num_steps=num_steps,
            C=C,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
            normalize=normalize,
            num_active=num_active,
        )


class DataModelFactory(Factory):
    data_model: Dict[str, Any] = {
        "cowan": CowanDynamics,
        "degree": DegreeDynamics,
        "glauber": GlauberDynamics,
        "sis": SISDynamics,
    }
    labeled_data_model: Dict[str, Any] = {
        "cowan": BlockLabeledCowanDynamics,
        "degree": BlockLabeledDegreeDynamics,
        "glauber": BlockLabeledGlauberDynamics,
        "sis": BlockLabeledSISDynamics,
    }
    nested_data_model: Dict[str, Any] = {
        "cowan": NestedBlockLabeledCowanDynamics,
        "degree": NestedBlockLabeledDegreeDynamics,
        "glauber": NestedBlockLabeledGlauberDynamics,
        "sis": NestedBlockLabeledSISDynamics,
    }

    @classmethod
    def build(cls, config: Config, constructors=None) -> Any:
        constructors = (
            DataModelFactory.data_model if constructors is None else constructors
        )
        if config.unmet_requirements():
            raise MissingRequirementsError(config)
        options = {
            k[6:]: getattr(cls, k) for k in cls.__dict__.keys() if k[:6] == "build_"
        }
        name = config.name
        if name in options:
            return options[name](config, constructors[name])
        else:
            raise OptionError(actual=name, expected=list(options.keys()))

    @classmethod
    def build_labeled(cls, config: Config):
        return cls.build(config, DataModelFactory.labeled_data_model)

    @classmethod
    def build_nested(cls, config: Config):
        return cls.build(config, DataModelFactory.nested_data_model)

    @staticmethod
    def build_glauber(config: DataModelConfig, constructor=GlauberDynamics):
        return constructor(
            config.num_steps,
            config.coupling,
            config.auto_activation_prob,
            config.auto_deactivation_prob,
            config.normalize,
            config.num_active,
        )

    @staticmethod
    def build_sis(config: DataModelConfig, constructor=SISDynamics):
        return constructor(
            config.num_steps,
            config.infection_prob,
            config.recovery_prob,
            config.auto_activation_prob,
            config.auto_deactivation_prob,
            config.normalize,
            config.num_active,
        )

    @staticmethod
    def build_cowan(config: DataModelConfig, constructor=CowanDynamics):
        return constructor(
            config.num_steps,
            config.nu,
            config.a,
            config.mu,
            config.eta,
            config.auto_activation_prob,
            config.auto_deactivation_prob,
            config.normalize,
            config.num_active,
        )

    @staticmethod
    def build_degree(config: DataModelConfig, constructor=DegreeDynamics):
        return constructor(config.num_steps, config.C)
