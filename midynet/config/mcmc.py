from __future__ import annotations
from typing import Any
from _midynet.proposer.block_proposer import BlockGenericProposer
from _midynet import mcmc
from .factory import Factory, OptionError, MissingRequirementsError
from .random_graph import RandomGraphConfig
from .proposer import EdgeProposerFactory, BlockProposerFactory
from .wrapper import Wrapper

__all__ = ("RandomGraphMCMCFactory", "MCMCVerboseFactory")


class RandomGraphMCMCFactory(Factory):
    @classmethod
    def build(cls, config: RandomGraphConfig) -> Any:
        if config.unmet_requirements():
            raise MissingRequirementsError(config)
        edge_proposer = EdgeProposerFactory.build(config.edge_proposer)
        block_proposer = (
            BlockProposerFactory.build(config.block_proposer)
            if "block_proposer" in config
            else BlockGenericProposer()
        )
        mcmc_model = mcmc.RandomGraphMCMC(edge_proposer, block_proposer)
        return Wrapper(
            mcmc_model,
            edge_proposer=edge_proposer,
            block_proposer=block_proposer,
        )


class MCMCVerboseFactory(Factory):
    @classmethod
    def build(cls, name: str, **kwargs) -> Any:
        options = {
            k[6:]: getattr(cls, k)
            for k in cls.__dict__.keys()
            if k[:6] == "build_"
        }
        if name in options:
            return options[name](**kwargs)
        else:
            raise OptionError(actual=name, expected=list(options.keys()))

    @classmethod
    def build_console(cls, types=None):
        callbacks = cls.collect_types(types)
        return Wrapper(
            mcmc.callbacks.VerboseToConsole(callbacks), callbacks=callbacks
        )

    @classmethod
    def build_file(cls, types=None, filename="./verbose.vb"):
        callbacks = cls.collect_types(types)
        return Wrapper(
            mcmc.callbacks.VerboseToFile(filename, callbacks),
            callbacks=callbacks,
        )

    @classmethod
    def collect_types(cls, types=None):
        verbose = []
        if types is None:
            verbose = [
                getattr(cls, k)()
                for k in cls.__dict__.keys()
                if k[:6] == "build_"
                and k != "build_file"
                and k != "build_console"
            ]
        else:
            verbose = [
                getattr(cls, k)()
                for k in cls.__dict__.keys()
                if k[:6] == "build_" and k[:6] in types
            ]
        return verbose

    @staticmethod
    def build_failure_counter():
        return mcmc.callbacks.FailureCounterVerbose()

    @staticmethod
    def build_success_counter():
        return mcmc.callbacks.SuccessCounterVerbose()

    @staticmethod
    def build_timer():
        return mcmc.callbacks.TimerVerbose()

    @staticmethod
    def build_mean_ratio():
        return mcmc.callbacks.MeanLogJointRatioVerbose()

    @staticmethod
    def build_max_ratio():
        return mcmc.callbacks.MaximumLogJointRatioVerbose()

    @staticmethod
    def build_min_ratio():
        return mcmc.callbacks.MinimumLogJointRatioVerbose()


if __name__ == "__main__":
    pass
