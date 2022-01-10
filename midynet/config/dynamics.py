from .config import Config
from .factory import Factory
from .wrapper import Wrapper
from _midynet import dynamics


class IsingGlauberDynamicsConfig(Config):
    requirements: set[str] = {"num_steps", "coupling"}

    @classmethod
    def default(cls, num_steps: int = 100, coupling: float = 1.0):
        return cls(name="default", num_steps=num_steps, coupling=coupling)


class SISDynamicsConfig(Config):
    requirements: set[str] = {"num_steps", "infection_prob", "recovery_prob"}

    @classmethod
    def default(
        cls,
        num_steps: int = 100,
        infection_prob: float = 0.5,
        recovery_prob: float = 0.5,
    ):
        return cls(
            name="default",
            num_steps=num_steps,
            infection_prob=infection_prob,
            recovery_prob=recovery_prob,
        )


class CowanDynamicsConfig(Config):
    requirements: set[str] = {"num_steps", "nu", "a", "mu", "eta"}

    @classmethod
    def default(
        cls,
        num_steps: int = 100,
        nu: float = 7.0,
        a: float = 1.0,
        mu: float = 1.0,
        eta: float = 0.5,
    ):
        return cls(
            name="default",
            num_steps=num_steps,
            nu=nu,
            a=a,
            mu=mu,
            eta=eta,
        )


class DegreeDynamicsConfig(Config):
    requirements: set[str] = {"num_steps", "C"}

    @classmethod
    def default(cls, num_steps: int = 100, C: float = 1.0):
        return cls(name="default", num_steps=num_steps, C=C)


class IsingGlauberDynamicsFactory(Factory):
    @staticmethod
    def build_default(config: IsingGlauberDynamicsConfig):
        return dynamics.IsingGlauberDynamics(config.num_steps, config.coupling)


class SISDynamicsFactory(Factory):
    @staticmethod
    def build_default(config: SISDynamicsConfig):
        return dynamics.SISDynamics(
            config.num_steps, config.infection_prob, config.recovery_prob
        )


class CowanDynamicsFactory(Factory):
    @staticmethod
    def build_default(config: CowanDynamicsConfig):
        return dynamics.CowanDynamics(
            config.num_steps, config.nu, config.a, config.mu, config.eta
        )


class DegreeDynamicsFactory(Factory):
    @staticmethod
    def build_default(config: DegreeDynamicsConfig):
        return dynamics.DegreeDynamics(config.num_steps, config.C)


if __name__ == "__main__":
    pass
