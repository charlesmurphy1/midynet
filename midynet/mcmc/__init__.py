from __future__ import annotations
from midynet.wrapper import Wrapper as _Wrapper
from midynet import random_graph as _random_graph
from midynet import data as _data
from _midynet import mcmc as _mcmc

from . import callbacks

__all__ = (
    "callbacks",
    "MCMCVerbose",
    "MCMCWrapper",
    "GraphReconstructionMCMC",
    "PartitionMCMC",
)


class MCMCVerbose(_Wrapper):
    def __init__(self, types=None):
        callbacks = self.collect_types(types)
        return _Wrapper(callbacks.VerboseToConsole(callbacks), callbacks=callbacks)

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
        return callbacks.FailureCounterVerbose()

    @staticmethod
    def build_success_counter():
        return callbacks.SuccessCounterVerbose()

    @staticmethod
    def build_timer():
        return callbacks.TimerVerbose()

    @staticmethod
    def build_mean_ratio():
        return callbacks.MeanLogJointRatioVerbose()

    @staticmethod
    def build_max_ratio():
        return callbacks.MaximumLogJointRatioVerbose()

    @staticmethod
    def build_min_ratio():
        return callbacks.MinimumLogJointRatioVerbose()


class MCMCWrapper(_Wrapper):
    def __init__(self, mcmc: _mcmc.MCMC, verbose=0, **kwargs):
        if verbose == 1:
            v = MCMCVerbose()
            self.insert_callback(v.wrap)
        elif issubclass(verbose.__class__, callbacks.VerboseDisplay):
            v = verbose
            self.insert_callback(v)
        else:
            v = None
        super().__init__(mcmc, verbose=v, **kwargs)


class GraphReconstructionMCMC(MCMCWrapper):
    def __init__(
        self,
        data_model: Union[_data.DataModel, _Wrapper],
        graph_prior: Union[_random_graph.RandomGraph, _Wrapper] = None,
        verbose=0,
        **kwargs,
    ):
        if graph_prior is None:
            graph_prior = data_model.get_graph_prior()
        else:
            if issubclass(graph_prior.__class__, _Wrapper):
                graph_wrapper = graph_prior.wrap
            else:
                graph_wrapper = graph_prior

            data_model.set_graph_prior(graph_wrapper)
        nested = False

        if issubclass(data_model.__class__, _Wrapper):
            data_wrapper = data_model.wrap
        else:
            data_wrapper = data_model

        if issubclass(data_wrapper.__class__, _data.DataModel):
            mcmc = _mcmc._GraphReconstructionMCMC(data_wrapper, **kwargs)
        elif issubclass(data_wrapper.__class__, _data.BlockLabeledDataModel):
            mcmc = _mcmc._BlockLabeledGraphReconstructionMCMC(data_wrapper, **kwargs)
        elif issubclass(data_wrapper.__class__, _data.NestedBlockLabeledDataModel):
            mcmc = _mcmc._NestedBlockLabeledGraphReconstructionMCMC(
                data_wrapper, **kwargs
            )
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


class PartitionReconstructionMCMC(MCMCWrapper):
    def __init__(
        self, graph_model: _random_graph.RandomGraph, verbose: int = 0, **kwargs
    ):
        nested = False
        if issubclass(graph_model.__class__, _Wrapper):
            graph_wrapper = graph_model.wrap
        else:
            graph_wrapper = graph_model
        if issubclass(graph_wrapper.__class__, _random_graph.BlockLabeledRandomGraph):
            mcmc = _mcmc._BlockLabelMCMC(graph_wrapper, **kwargs)
        elif issubclass(
            graph_wrapper.__class__, _random_graph.NestedBlockLabeledRandomGraph
        ):
            mcmc = _mcmc._NestedBlockLabelMCMC(graph_wrapper, **kwargs)
            nested = True
        else:
            raise TypeError(f"BlockMCMC: wrong type `{graph_wrapper.__class__}`.")
        super().__init__(
            mcmc, verbose=verbose, graph_model=graph_model, nested=nested, **kwargs
        )
