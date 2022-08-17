from __future__ import annotations

from midynet.wrapper import Wrapper as _Wrapper
from . import dynamics
from _midynet.data import DataModel, BlockLabeledDataModel, NestedBlockLabeledDataModel
from _midynet import data as _data
from midynet import random_graph as _rg

__all__ = (
    "DataModel",
    "BlockLabeledDataModel",
    "NestedBlockLabeledDataModel",
    "DataModelWrapper" "dynamics",
)


def get_graph_prior_type(graph_prior: Union[_rg.RandomGraph, _Wrapper]):
    if issubclass(graph_prior.__class__, _Wrapper):
        wrapped = graph_prior.wrap
    else:
        wrapped = graph_prior

    if issubclass(wrapped.__class__, _rg.NestedBlockLabeledRandomGraph):
        return wrapped, "nested"
    elif issubclass(wrapped.__class__, _rg.BlockLabeledRandomGraph):
        return wrapped, "labeled"
    return wrapped, "normal"


class DataModelWrapper(_Wrapper):
    constructors = {}

    def __init__(self, graph_prior: RandomGraph = None, **kwargs):
        graph_prior = (
            _rg.ErdosRenyiModel(100, 250) if graph_prior is None else graph_prior
        )
        wrapped, dtype = get_graph_prior_type(graph_prior)
        if len(self.constructors) == 0:
            raise ValueError("`constructors` must not be empty.")
        data_model = self.constructors[dtype](wrapped, **kwargs)
        super().__init__(data_model, graph_prior=graph_prior, params=kwargs)

    def set_graph_prior(self, graph_prior: Union[RandomGraph, _Wrapper]):
        wrapped, dtype = get_graph_prior_type(graph_prior)
        self.__wrapped__ = self.constructors[dtype](wrapped, **self.others["params"])
        self.__others__["graph_prior"] = graph_prior
