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
    verbose_types = [
        "failure_counter",
        "success_counter",
        "mean_ratio",
        "max_ratio",
        "min_ratio",
        "timer",
    ]

    def __init__(self, types=None):
        cbs = self.collect_types(types)
        super().__init__(_mcmc.callbacks.VerboseToConsole(cbs), callbacks=cbs)

    def collect_types(self, types=None):
        verbose = []
        if types is None:
            verbose_collection = [
                getattr(self, "build_" + k)() for k in self.verbose_types
            ]
        else:
            verbose_collection = [
                getattr(self, k)() for k in types if k in self.verbose_types
            ]
        return verbose_collection

    def build_failure_counter(self):
        return callbacks.FailureCounterVerbose()

    def build_success_counter(self):
        return callbacks.SuccessCounterVerbose()

    def build_timer(self):
        return callbacks.TimerVerbose()

    def build_mean_ratio(self):
        return callbacks.MeanLogJointRatioVerbose()

    def build_max_ratio(self):
        return callbacks.MaximumLogJointRatioVerbose()

    def build_min_ratio(self):
        return callbacks.MinimumLogJointRatioVerbose()


class MCMCWrapper(_Wrapper):
    def __init__(self, mcmc: _mcmc.MCMC, verbose=0, **kwargs):
        if verbose == 1:
            v = MCMCVerbose()
            mcmc.insert_callback("verbose", v.wrap)
        elif issubclass(verbose.__class__, _mcmc.callbacks.VerboseDisplay):
            v = verbose
            mcmc.insert_callback("verbose", v)
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
