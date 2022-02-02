from _midynet import dynamics

from .config import Config
from .factory import Factory

__all__ = ("DynamicsConfig", "DynamicsFactory")


class DynamicsConfig(Config):
    requirements: set[str] = {"num_steps"}

    def set_coupling(self, coupling):
        if self.name == "sis":
            self.set_value("infection_prob", coupling)
        elif self.name == "ising":
            self.set_value("coupling", coupling)
        elif self.name == "cowan":
            self.set_value("nu", coupling)
        else:
            message = (
                f"Invalid entry {dynamics} for dynamics,"
                + "expected ['sis', 'ising', 'cowan']."
            )
            raise ValueError(message)

    @classmethod
    def ising(
        cls,
        num_steps: int = 100,
        coupling: float = 1.0,
        normalize: bool = True,
    ):
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
        infection_prob: float = 0.5,
        recovery_prob: float = 0.5,
        auto_infection_prob: float = 1e-6,
        normalize: bool = True,
    ):
        return cls(
            name="sis",
            num_steps=num_steps,
            infection_prob=infection_prob,
            recovery_prob=recovery_prob,
            auto_infection_prob=auto_infection_prob,
            normalize=normalize,
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
    ):
        return cls(
            name="cowan",
            num_steps=num_steps,
            nu=nu,
            a=a,
            mu=mu,
            eta=eta,
            normalize=normalize,
        )

    @classmethod
    def degree(cls, num_steps: int = 100, C: float = 1.0):
        return cls(name="degree", num_steps=num_steps, C=C)


class DynamicsFactory(Factory):
    @staticmethod
    def build_ising(config: DynamicsConfig):
        return dynamics.IsingGlauberDynamics(
            config.num_steps, config.coupling, config.normalize
        )

    @staticmethod
    def build_sis(config: DynamicsConfig):
        return dynamics.SISDynamics(
            config.num_steps,
            config.infection_prob,
            config.recovery_prob,
            config.auto_infection_prob,
            config.normalize,
        )

    @staticmethod
    def build_cowan(config: DynamicsConfig):
        return dynamics.CowanDynamics(
            config.num_steps,
            config.nu,
            config.a,
            config.mu,
            config.eta,
            config.normalize,
        )

    @staticmethod
    def build_degree(config: DynamicsConfig):
        return dynamics.DegreeDynamics(config.num_steps, config.C)


if __name__ == "__main__":
    pass
