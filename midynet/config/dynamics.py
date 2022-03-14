from __future__ import annotations
from typing import Set
from _midynet.dynamics import (
    SISDynamics,
    IsingGlauberDynamics,
    CowanDynamics,
    DegreeDynamics,
)
from .config import Config
from .factory import Factory

__all__ = ("DynamicsConfig", "DynamicsFactory")


class DynamicsConfig(Config):
    requirements: Set[str] = {"num_steps"}

    def set_coupling(self, coupling: float) -> None:
        if self.name == "sis":
            self.set_value("infection_prob", coupling)
        elif self.name == "ising":
            self.set_value("coupling", coupling)
        elif self.name == "cowan":
            self.set_value("nu", coupling)
        else:
            message = (
                f"Invalid entry {self.name} for dynamics,"
                + "expected ['sis', 'ising', 'cowan']."
            )
            raise ValueError(message)

    def get_coupling(self):
        if self.name == "sis":
            return self.infection_prob
        elif self.name == "ising":
            return self.coupling
        elif self.name == "cowan":
            return self.nu
        else:
            message = (
                f"Invalid entry {self.name} for dynamics,"
                + "expected ['sis', 'ising', 'cowan']."
            )
            raise ValueError(message)

    @classmethod
    def ising(
        cls,
        num_steps: int = 100,
        coupling: float = 1.0,
        normalize: bool = True,
    ) -> DynamicsConfig:
        return cls(
            name="ising",
            num_steps=num_steps,
            coupling=coupling,
            normalize=normalize,
        )

    @classmethod
    def sis(
        cls,
        num_steps: int = 100,
        infection_prob: float = 0.1,
        recovery_prob: float = 0.1,
        auto_infection_prob: float = 1e-4,
        normalize: bool = True,
        num_active: int = 1,
    ) -> DynamicsConfig:
        return cls(
            name="sis",
            num_steps=num_steps,
            infection_prob=infection_prob,
            recovery_prob=recovery_prob,
            auto_infection_prob=auto_infection_prob,
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
            normalize=normalize,
            num_active=num_active,
        )

    @classmethod
    def degree(cls, num_steps: int = 100, C: float = 1.0) -> DynamicsConfig:
        return cls(name="degree", num_steps=num_steps, C=C)


class DynamicsFactory(Factory):
    @staticmethod
    def build_ising(config: DynamicsConfig) -> IsingGlauberDynamics:
        return IsingGlauberDynamics(
            config.num_steps, config.coupling, config.normalize
        )

    @staticmethod
    def build_sis(config: DynamicsConfig) -> SISDynamics:
        return SISDynamics(
            config.num_steps,
            config.infection_prob,
            config.recovery_prob,
            config.auto_infection_prob,
            config.normalize,
            config.num_active,
        )

    @staticmethod
    def build_cowan(config: DynamicsConfig) -> CowanDynamics:
        return CowanDynamics(
            config.num_steps,
            config.nu,
            config.a,
            config.mu,
            config.eta,
            config.normalize,
            config.num_active,
        )

    @staticmethod
    def build_degree(config: DynamicsConfig) -> DegreeDynamics:
        return DegreeDynamics(config.num_steps, config.C)
