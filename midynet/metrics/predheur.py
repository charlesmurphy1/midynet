import numpy as np
import networkx as nx

from typing import Dict, Any, List, Callable
from basegraph import core as bg
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)
from itertools import product

from midynet.utility.convert import convert_basegraph_to_networkx
from midynet.config import Config, GraphFactory, DataModelFactory
from graphinf.utility import seed as gi_seed
from .metrics import ExpectationMetrics
from .multiprocess import Expectation


class FeatureExtractor:
    __collection__ = dict()

    def __init__(self, features: List[str] = None):
        self.features = features if features is not None else []
        if self.features == "all":
            self.features = list(self.__collection__.keys())
        if isinstance(self.features, str):
            self.features = (
                list(self.__collection__.keys())
                if self.features == "all"
                else [self.features]
            )
        for f in self.features:
            if f not in self.__collection__:
                msg = f"feature '{f}' is not in collection: available features are '{self.__collection__}'."
                raise ValueError(msg)

    def preprocess(self, data: Any) -> Any:
        return data

    def postprocess(self, features: Any) -> Any:
        return features

    def __call__(self, data: Any) -> Dict[str, np.ndarray]:
        data = self.preprocess(data)
        out = dict()
        for f in self.features:
            out[f] = self.postprocess(self.__collection__[f](data))
        return out


class GraphFeatureExtractor(FeatureExtractor):
    __collection__: Dict[str, Callable[[nx.Graph], Dict[int, float]]] = dict(
        degree=lambda g: dict(nx.degree(g)),
        kcore=nx.core_number,
        eigen=nx.eigenvector_centrality,
        closeness=nx.closeness_centrality,
        clustering=nx.clustering,
        pagerank=nx.pagerank,
    )

    def preprocess(self, graph: bg.UndirectedMultigraph) -> nx.Graph:
        graph = convert_basegraph_to_networkx(graph)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        return graph

    def postprocess(self, node_features: Dict[int, float]) -> np.ndarray:
        return np.array(
            [v for _, v in sorted(node_features.items(), key=lambda x: x[0])]
        )


class StateFeatureExtractor(FeatureExtractor):
    __collection__: Dict[str, Callable[[np.ndarray], np.ndarray]] = dict(
        mean=lambda x: np.mean(x, axis=-1),
        std=lambda x: np.std(x, axis=-1),
        var=lambda x: np.var(x, axis=-1),
        last=lambda x: x[:, -1],
    )


class PredictionHeuristics(Expectation):
    def __init__(self, config: Config, **kwargs):
        self.params = config.dict
        super().__init__(**kwargs)

    def score(self, g_feats, x_feats) -> Dict[str, float]:
        raise NotImplementedError()

    def func(self, seed: int) -> float:
        gi_seed(seed)
        config = Config.from_dict(self.params)
        graph_model = GraphFactory.build(config.prior)
        data_model = DataModelFactory.build(config.data_model)
        data_model.set_graph_prior(graph_model)
        if config.target != "None":
            g0 = GraphFactory.build(config.target)
        else:
            g0 = graph_model.state()
        x0 = data_model.random_state(config.data_model.get("n_active", -1))
        data_model.set_graph(g0)
        data_model.sample_state(x0)
        timeseries = np.array(data_model.past_states())
        g_feats = GraphFeatureExtractor(config.metrics.get("graph_features"))(
            g0
        )
        x_feats = StateFeatureExtractor(config.metrics.get("state_features"))(
            timeseries
        )
        return self.score(g_feats, x_feats)


class LinearRegressionHeuristics(PredictionHeuristics):
    def score(
        self, g_feats: Dict[str, np.ndarray], x_feats: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        model = LinearRegression()
        inputs = np.concatenate(
            [v[:, np.newaxis] for v in g_feats.values()], axis=1
        )
        targets = np.concatenate(
            [v[:, np.newaxis] for v in x_feats.values()], axis=1
        )
        model.fit(inputs, targets)
        return dict(determination=model.score(inputs, targets))


class LinearRegressionHeuristicsMetrics(ExpectationMetrics):
    shortname = "linregheur"
    keys = ["determination"]
    expectation_factory = LinearRegressionHeuristics


class MutualInformationHeuristics(PredictionHeuristics):
    def score(self, g_feats: np.ndarray, x_feats: np.ndarray):
        g_keys = g_feats.keys()
        x_keys = x_feats.keys()
        inputs = np.concatenate(
            [g_feats[k][:, np.newaxis] for k in g_keys], axis=-1
        )
        out = {}
        for xk in x_keys:
            x = x_feats[xk]
            mutual_info = (
                mutual_info_regression
                if x.dtype == float
                else mutual_info_classif
            )
            mi = {k: v for k, v in zip(g_keys, mutual_info(inputs, x))}
            for gk, v in mi.items():
                out["mi_" + xk + "_" + gk] = v
        return out


class MutualInformationHeuristicsMetrics(ExpectationMetrics):
    shortname = "miheur"
    keys = [
        f"mi_{xk}_{gk}"
        for xk, gk in product(
            StateFeatureExtractor.__collection__.keys(),
            GraphFeatureExtractor.__collection__.keys(),
        )
    ]
    expectation_factory = MutualInformationHeuristics
