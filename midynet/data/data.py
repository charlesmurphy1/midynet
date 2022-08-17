from typing import Union

from _midynet.data import DataModel
from midynet.wrapper import Wrapper
from midynet.random_graph import (
    RandomGraph,
    BlockLabeledRandomGraph,
    NestedBlockLabeledRandomGraph,
    ErdosRenyiModel,
)

__all__ = ("DataModel", "DataModelWrapper")


def get_graph_prior_type(graph_prior: Union[RandomGraph, Wrapper]):
    if issubclass(graph_prior.__class__, Wrapper):
        wrapped = graph_prior.wrap
    else:
        wrapped = graph_prior

    if issubclass(wrapped.__class__, NestedBlockLabeledRandomGraph):
        return wrapped, "nested"
    elif issubclass(wrapped.__class__, BlockLabeledRandomGraph):
        return wrapped, "labeled"
    return wrapped, "normal"


class DataModelWrapper(Wrapper):
    constructors = {}

    def __init__(self, graph_prior: RandomGraph = None, **kwargs):
        graph_prior = ErdosRenyiModel(100, 250) if graph_prior is None else graph_prior
        wrapped, dtype = get_graph_prior_type(graph_prior)
        data_model = self.constructors[dtype](wrapped, **kwargs)
        super().__init__(data_model, graph_prior=graph_prior, params=kwargs)

    def set_graph_prior(self, graph_prior: Union[RandomGraph, Wrapper]):
        wrapped, dtype = get_graph_prior_type(graph_prior)
        self.__wrapped__ = self.constructors[dtype](wrapped, **self.others["params"])
        self.__others__["graph_prior"] = graph_prior
