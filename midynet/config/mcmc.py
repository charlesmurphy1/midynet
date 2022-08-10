from __future__ import annotations
from typing import Any

from _midynet.random_graph import (
    RandomGraph,
    BlockLabeledRandomGraph,
    NestedBlockLabeledRandomGraph,
)
from _midynet.data import DataModel, BlockLabeledDataModel, NestedBlockLabeledDataModel
from _midynet.mcmc import (
    GraphReconstructionMCMC,
    BlockLabeledGraphReconstructionMCMC,
    NestedBlockLabeledGraphReconstructionMCMC,
    BlockLabelMCMC,
    NestedBlockLabelMCMC,
)
from _midynet.mcmc import callbacks as cb
from .config import Config
from .factory import Factory, OptionError, MissingRequirementsError
from .wrapper import Wrapper

__all__ = ("ReconstructionMCMC", "PartitionMCMC", "MCMCVerboseFactory")


class ReconstructionMCMC(Wrapper):
    def __init__(self, data_model, graph_prior, **kwargs):
        data_model.set_graph_prior(graph_prior)
        if issubclass(data_model.__class__, DataModel):
            mcmc = GraphReconstructionMCMC(data_model, **kwargs)
        elif issubclass(data_model.__class__, BlockLabeledDataModel):
            mcmc = BlockLabeledGraphReconstructionMCMC(data_model, **kwargs)
        elif issubclass(data_model.__class__, NestedBlockLabeledDataModel):
            mcmc = NestedBlockLabeledGraphReconstructionMCMC(data_model, **kwargs)
        else:
            raise TypeError(f"ReconstructionMCMC: wrong type `{data_model.__class__}`.")
        super().__init__(mcmc, data_model=data_model, graph_prior=graph_prior, **kwargs)


class PartitionMCMC(Wrapper):
    def __init__(self, graph, **kwargs):
        if issubclass(graph, BlockLabeledRandomGraph):
            mcmc = BlockLabelMCMC(graph, **kwargs)
        elif issubclass(graph, NestedBlockLabeledRandomGraph):
            mcmc = NestedBlockLabelMCMC(graph, **kwargs)
        else:
            raise TypeError(f"BlockMCMC: wrong type `{graph.__class__}`.")
        super().__init__(mcmc, graph=graph, **kwargs)


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
