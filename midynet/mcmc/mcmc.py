from __future__ import annotations
from typing import Any

from midynet.wrapper import Wrapper
from _midynet.random_graph import (
    RandomGraph,
    BlockLabeledRandomGraph,
    NestedBlockLabeledRandomGraph,
)
from _midynet.data import DataModel, BlockLabeledDataModel, NestedBlockLabeledDataModel
from _midynet.mcmc import (
    MCMC,
    _GraphReconstructionMCMC,
    _BlockLabeledGraphReconstructionMCMC,
    _NestedBlockLabeledGraphReconstructionMCMC,
    _BlockLabelMCMC,
    _NestedBlockLabelMCMC,
)
from _midynet.mcmc import callbacks as cb


__all__ = (
    "MCMCVerbose",
    "MCMCWrapper",
    "GraphReconstructionMCMC",
    "PartitionMCMC",
)


class MCMCVerbose(Wrapper):
    def __init__(self, types=None):
        callbacks = self.collect_types(types)
        return Wrapper(cb.VerboseToConsole(callbacks), callbacks=callbacks)

    @staticmethod
    def collect_types(types=None):
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


class MCMCWrapper(Wrapper):
    def __init__(self, mcmc: MCMC, verbose=0, **kwargs):
        if verbose == 1:
            v = MCMCVerbose()
            self.insert_callback(v.wrap)
        elif issubclass(verbose.__class__, cb.VerboseDisplay):
            v = verbose
            self.insert_callback(v)
        else:
            v = None
        super().__init__(mcmc, verbose=v, **kwargs)


class GraphReconstructionMCMC(MCMCWrapper):
    def __init__(
        self,
        data_model: Union[DataModel, Wrapper],
        graph_prior: Union[RandomGraph, Wrapper] = None,
        verbose=0,
        **kwargs,
    ):
        if graph_prior is None:
            graph_prior = data_model.get_graph_prior()
        else:
            if issubclass(graph_prior.__class__, Wrapper):
                graph_wrapper = graph_prior.wrap
            else:
                graph_wrapper = graph_prior

            data_model.set_graph_prior(graph_wrapper)
        nested = False

        if issubclass(data_model.__class__, Wrapper):
            data_wrapper = data_model.wrap
        else:
            data_wrapper = data_model

        if issubclass(data_wrapper.__class__, DataModel):
            mcmc = _GraphReconstructionMCMC(data_wrapper, **kwargs)
        elif issubclass(data_wrapper.__class__, BlockLabeledDataModel):
            mcmc = _BlockLabeledGraphReconstructionMCMC(data_wrapper, **kwargs)
        elif issubclass(data_wrapper.__class__, NestedBlockLabeledDataModel):
            mcmc = _NestedBlockLabeledGraphReconstructionMCMC(data_wrapper, **kwargs)
            nested = True
        else:
            raise TypeError(
                f"ReconstructionMCMC: wrong type `{data_wrapper.__class__}`."
            )

        super().__init__(
            mcmc,
            data_model=data_model,
            graph_prior=graph_prior,
            nested=nested,
            verbose=verbose,
            **kwargs,
        )


class PartitionMCMC(MCMCWrapper):
    def __init__(self, graph_model: RandomGraph, verbose: int = 0, **kwargs):
        nested = False
        if issubclass(graph_model.__class__, Wrapper):
            graph_wrapper = graph_model.wrap
        else:
            graph_wrapper = graph_model
        if issubclass(graph_wrapper.__class__, BlockLabeledRandomGraph):
            mcmc = _BlockLabelMCMC(graph_wrapper, **kwargs)
        elif issubclass(graph_wrapper.__class__, NestedBlockLabeledRandomGraph):
            mcmc = _NestedBlockLabelMCMC(graph_wrapper, **kwargs)
            nested = True
        else:
            raise TypeError(f"BlockMCMC: wrong type `{graph_wrapper.__class__}`.")
        super().__init__(
            mcmc, verbose=verbose, graph_model=graph_model, nested=nested, **kwargs
        )
