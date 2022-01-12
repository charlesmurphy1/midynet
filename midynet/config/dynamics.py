from .config import Config
from .factory import Factory
from .wrapper import Wrapper
from _midynet import dynamics

__all__ = ["DynamicsConfig", "DynamicsFactory"]


class DynamicsConfig(Config):
    requirements: set[str] = {"num_steps"}

    @classmethod
    def ising(cls, num_steps: int = 100, coupling: float = 1.0):
        return cls(name="ising", num_steps=num_steps, coupling=coupling)

    @classmethod
    def sis(
        cls,
        num_steps: int = 100,
        infection_prob: float = 0.5,
        recovery_prob: float = 0.5,
    ):
        return cls(
            name="sis",
            num_steps=num_steps,
            infection_prob=infection_prob,
            recovery_prob=recovery_prob,
        )

    @classmethod
    def cowan(
        cls,
        num_steps: int = 100,
        nu: float = 7.0,
        a: float = 1.0,
        mu: float = 1.0,
        eta: float = 0.5,
    ):
        return cls(
            name="cowan",
            num_steps=num_steps,
            nu=nu,
            a=a,
            mu=mu,
            eta=eta,
        )

    @classmethod
    def degree(cls, num_steps: int = 100, C: float = 1.0):
        return cls(name="degree", num_steps=num_steps, C=C)


class DynamicsFactory(Factory):
    @staticmethod
    def build_ising(config: DynamicsConfig):
        return dynamics.IsingGlauberDynamics(config.num_steps, config.coupling)

    @staticmethod
    def build_sis(config: DynamicsConfig):
        return dynamics.SISDynamics(
            config.num_steps, config.infection_prob, config.recovery_prob
        )

    @staticmethod
    def build_cowan(config: DynamicsConfig):
        return dynamics.CowanDynamics(
            config.num_steps, config.nu, config.a, config.mu, config.eta
        )

    @staticmethod
    def build_degree(config: DynamicsConfig):
        return dynamics.DegreeDynamics(config.num_steps, config.C)


if __name__ == "__main__":
    pass
