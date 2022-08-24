from __future__ import annotations

from midynet.wrapper import Wrapper as _Wrapper
from midynet.random_graph import (
    RandomGraphWrapper as _RandomGraphWrapper,
    ErdosRenyiModel as _ErdosRenyiModel,
)
from . import dynamics
from _midynet.data import DataModel, BlockLabeledDataModel, NestedBlockLabeledDataModel
from _midynet import data as _data

__all__ = (
    "DataModel",
    "BlockLabeledDataModel",
    "NestedBlockLabeledDataModel",
    "DataModelWrapper" "dynamics",
)


class DataModelWrapper(_Wrapper):
    constructors = {}

    def __init__(self, graph_prior: _RandomGraphWrapper = None, **kwargs):
        if len(self.constructors) == 0:
            raise ValueError("`constructors` must not be empty.")
        graph_prior = _ErdosRenyiModel(100, 250) if graph_prior is None else graph_prior
        self.labeled = graph_prior.labeled
        self.nested = graph_prior.nested
        data_model = self.constructors[self.dtype](graph_prior.wrap, **kwargs)
        super().__init__(data_model, graph_prior=graph_prior, params=kwargs)

    @property
    def dtype(self):
        if self.nested:
            return "nested"
        elif self.labeled:
            return "labeled"
        return "normal"

    def set_graph_prior(self, graph_prior: _RandomGraphWrapper):
        self.labeled = graph_prior.labeled
        self.nested = graph_prior.nested
        self.__wrapped__ = self.constructors[self.dtype](
            graph_prior.wrap, **self.others["params"]
        )
        self.__others__["graph_prior"] = graph_prior

    def get_graph_prior(self):
        return self.__others__["graph_prior"]
