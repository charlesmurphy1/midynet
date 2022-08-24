from __future__ import annotations
from midynet.wrapper import Wrapper as _Wrapper
from midynet.random_graph import RandomGraphWrapper as _RandomGraphWrapper
from midynet.data import DataModelWrapper as _DataModelWrapper
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
        data_model: _DataModelWrapper,
        graph_prior: _RandomGraphWrapper = None,
        verbose=0,
        **kwargs,
    ):
        if graph_prior is None:
            graph_prior = data_model.get_graph_prior()
        else:
            data_model.set_graph_prior(graph_prior)

        if data_model.nested:
            mcmc = _mcmc._NestedBlockLabeledGraphReconstructionMCMC(
                data_model.wrap, **kwargs
            )
        elif data_model.labeled:
            mcmc = _mcmc._BlockLabeledGraphReconstructionMCMC(data_model.wrap, **kwargs)
        else:
            mcmc = _mcmc._GraphReconstructionMCMC(data_model.wrap, **kwargs)

        super().__init__(
            mcmc,
            data_model=data_model,
            graph_prior=graph_prior,
            verbose=verbose,
            **kwargs,
        )


class PartitionReconstructionMCMC(MCMCWrapper):
    def __init__(self, graph_model: RandomGraphWrapper, verbose: int = 0, **kwargs):
        if graph_model.nested:
            mcmc = _mcmc._NestedPartitionReconstructionMCMC(graph_model.wrap, **kwargs)
        elif graph_model.labeled:
            mcmc = _mcmc._PartitionReconstructionMCMC(graph_model.wrap, **kwargs)
        else:
            raise TypeError(
                f"PartitionReconstructionMCMC: wrong type `{graph_model.wrap.__class__}`."
            )
        super().__init__(mcmc, verbose=verbose, graph_model=graph_model, **kwargs)

    def get_labels(self):
        if self.graph_model.nested:
            return self.graph_model.get_nested_labels()
        return self.graph_model.get_labels()

    def set_labels(self, labels, reduce=False):
        if self.graph_model.nested:
            self.graph_model.set_nested_labels(labels, reduce)
        else:
            self.graph_model.set_labels(labels, reduce)
