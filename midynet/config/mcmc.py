from __future__ import annotations
from typing import Any

from _midynet.mcmc import (
    GraphReconstructionMCMC,
    BlockLabeledGraphReconstructionMCMC,
    BlockLabelMCMC,
)
from _midynet.mcmc import callbacks as cb
from .config import Config
from .factory import Factory, OptionError, MissingRequirementsError
from .proposer import EdgeProposerFactory, BlockProposerFactory
from .random_graph import RandomGraphFactory
from .dynamics import DynamicsFactory
from .wrapper import Wrapper

__all__ = ("MCMCFactory", "MCMCVerboseFactory")


class MCMCFactory(Factory):
    @staticmethod
    def build_reconstruction(config: ExperimentConfig):
        graph = RandomGraphFactory.build(config.graph)
        edge_proposer = EdgeProposerFactory.build(config.graph.edge_proposer)
        if not config.graph.labeled:
            dynamics = DynamicsFactory.build(config.dynamics)
            dynamics.set_graph_prior(graph.wrap)
            mcmc = GraphReconstructionMCMC(dynamics, edge_proposer)
            return Wrapper(
                mcmc, dynamics=dynamics, graph=graph, edge_proposer=edge_proposer
            )
        dynamics = DynamicsFactory.build_labeled(config.dynamics)
        dynamics.set_graph_prior(graph.wrap)
        block_proposer = BlockProposerFactory.build(config.graph.block_proposer)
        mcmc = BlockLabeledGraphReconstructionMCMC(
            dynamics, edge_proposer, block_proposer
        )
        return Wrapper(
            mcmc,
            dynamics=dynamics,
            graph=graph,
            edge_proposer=edge_proposer,
            block_proposer=block_proposer,
        )

    @staticmethod
    def build_community(config: RandomGraphConfig):
        graph = RandomGraphFactory.build(config)
        if config.graph.labeled:
            block_proposer = BlockProposerFactory.build(graph.block_proposer)
            mcmc = BlockLabelMCMC(dynamics, block_proposer)
            return Wrapper(mcmc, graph=graph, block_proposer=block_proposer)
        return None


class MCMCVerboseFactory(Factory):
    @classmethod
    def build(cls, name: str, **kwargs: Any) -> Any:
        options = {
            k[6:]: getattr(cls, k) for k in cls.__dict__.keys() if k[:6] == "build_"
        }
        if name in options:
            return options[name](**kwargs)
        else:
            raise OptionError(actual=name, expected=list(options.keys()))

    @classmethod
    def build_console(cls, types=None):
        callbacks = cls.collect_types(types)
        return Wrapper(cb.VerboseToConsole(callbacks), callbacks=callbacks)

    @classmethod
    def build_file(cls, types=None, filename="./verbose.vb"):
        callbacks = cls.collect_types(types)
        return Wrapper(
            cb.VerboseToFile(filename, callbacks),
            callbacks=callbacks,
        )

    @classmethod
    def collect_types(cls, types=None):
        verbose = []
        if types is None:
            verbose = [
                getattr(cls, k)()
                for k in cls.__dict__.keys()
                if k[:6] == "build_" and k != "build_file" and k != "build_console"
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
        return cb.FailureCounterVerbose()

    @staticmethod
    def build_success_counter():
        return cb.SuccessCounterVerbose()

    @staticmethod
    def build_timer():
        return cb.TimerVerbose()

    @staticmethod
    def build_mean_ratio():
        return cb.MeanLogJointRatioVerbose()

    @staticmethod
    def build_max_ratio():
        return cb.MaximumLogJointRatioVerbose()

    @staticmethod
    def build_min_ratio():
        return cb.MinimumLogJointRatioVerbose()


if __name__ == "__main__":
    pass
