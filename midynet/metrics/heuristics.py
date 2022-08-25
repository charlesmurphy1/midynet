from __future__ import annotations
import networkx as nx
import numpy as np
import time
import warnings

from midynet import utility
from midynet.config import Config, OptionError, RandomGraphFactory, DataModelFactory
from .metrics import Metrics
from .multiprocess import Expectation
from .statistics import Statistics
from dataclasses import dataclass, field
from netrd import reconstruction as _reconstruction
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def ignore_warnings(func):
    def wrapper(*args, **kwargs):
        np.seterr(invalid="ignore")
        warnings.filterwarnings("ignore")

        value = func(*args, **kwargs)

        np.seterr(invalid="ignore")
        warnings.filterwarnings("ignore")

        return value

    return wrapper


class ReconstructionHeuristicsMethod:
    def __init__(self):
        self.clear()

    @property
    def pred(self):
        return self.__results__["pred"]

    def fit(self, timeseries, **kwargs):
        raise NotImplementedError()

    def compare(self, true_graph, measures=None, **kwargs):
        if len(self.__results__) == 0:
            raise ValueError("`results` must not be empty.")

        measures = ["roc"] if measures is None else measures

        true = np.array(true_graph.get_adjacency_matrix())
        np.fill_diagonal(true, 0)
        true[true > 1] = 1
        for m in measures:
            if hasattr(self, "collect_" + m):
                self.__results__[m] = getattr(self, "collect_" + m)(true, self.pred)
            else:
                warnings.warn(
                    f"no collector named `{m}` has been found, proceeding anyway.",
                    RuntimeWarning,
                )
        return self.__results__.copy()

    def clear(self):
        self.__results__ = {}

    def normalize_weights(self, weights):
        if weights.min() == weights.max():
            return weights
        weights = (weights - weights.min()) / (weights.max() - weights.min())
        return weights

    def collect_confusion_matrix(self, true, pred, **kwargs):
        threshold = kwargs.get("threshold", norm_pred.mean())
        norm_pred = self.normalize_weights(pred).reshape(-1)
        true = true.reshape(-1)
        cm = confusion_matrix(true, (norm_pred > threshold).astype("float").reshape(-1))
        tn, fp, fn, tp = cm.ravel()

        return dict(threshold=threshold, tn=tn, fp=fp, fn=fn, tp=tp)

    def collect_roc(self, true, pred, **kwargs):
        pred[np.isnan(pred)] = -10
        pred = self.normalize_weights(pred).reshape(-1).astype("float")
        true = true.reshape(-1).astype("int")
        fpr, tpr, thresholds = roc_curve(true, pred)
        auc = roc_auc_score(true, pred)

        return dict(fpr=fpr, tpr=tpr, auc=auc, thresholds=thresholds)


class WeightbasedReconstructionHeuristicsMethod(ReconstructionHeuristicsMethod):
    def __init__(self, model, nanfill=None):
        self.model = model
        self.nanfill = 0 if nanfill is None else nanfill
        super().__init__()

    @ignore_warnings
    def fit(self, timeseries, **kwargs):
        self.clear()
        self.model.fit(timeseries.T, **kwargs)
        weights = self.model.results["weights_matrix"]
        weights[np.isnan(weights)] = self.nanfill
        np.fill_diagonal(weights, 0)
        self.__results__["pred"] = weights


class GraphbasedReconstructionHeuristicsMethod(ReconstructionHeuristicsMethod):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def fit(self, timeseries, **kwargs):
        self.clear()
        self.model.fit(timeseries.T, **kwargs)
        weights = nx.to_numpy_array(self.model.results["graph"])
        self.__results__["pred"] = weights


def get_heuristics_reconstructor(config):
    reconstructors = {
        "correlation": lambda: WeightbasedReconstructionHeuristicsMethod(
            _reconstruction.CorrelationMatrix()
        ),
        "granger_causality": lambda: WeightbasedReconstructionHeuristicsMethod(
            _reconstruction.GrangerCausality()
        ),
        "transfer_entropy": lambda: WeightbasedReconstructionHeuristicsMethod(
            _reconstruction.NaiveTransferEntropy()
        ),
        "graphical_lasso": lambda: WeightbasedReconstructionHeuristicsMethod(
            _reconstruction.GraphicalLasso()
        ),
        "mutual_information": lambda: WeightbasedReconstructionHeuristicsMethod(
            _reconstruction.MutualInformationMatrix()
        ),
        "partial_correlation": lambda: WeightbasedReconstructionHeuristicsMethod(
            _reconstruction.PartialCorrelationMatrix()
        ),
        "correlation_spanning_tree": lambda: GraphbasedReconstructionHeuristicsMethod(
            _reconstruction.CorrelationSpanningTree()
        ),
    }

    if config.method in reconstructors:
        return reconstructors[config.method]()
    else:
        raise OptionError(actual=config.method, expected=reconstructors.keys())


@dataclass
class ReconstructionHeuristics(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        utility.seed(seed)

        graph_model = RandomGraphFactory.build(self.config.graph_prior)
        data_model = DataModelFactory.build(self.config.data_model)
        data_model.set_graph_prior(graph_model)

        data_model.sample()
        timeseries = np.array(data_model.get_past_states()).T
        heuristics = get_heuristics_reconstructor(self.config.metrics.heuristics)
        heuristics.fit(timeseries)
        heuristics.compare(graph_model.get_state(), collectors=["roc"])

        return heuristics.__results__["roc"]["auc"]


class ReconstructionHeuristicsMetrics(Metrics):
    def eval(self, config: Config):
        heuristics_auc = ReconstructionHeuristics(
            config=config,
            num_procs=config.get_value("num_procs", 1),
            seed=config.get_value("seed", int(time.time())),
        )
        samples = heuristics_auc.compute(
            config.metrics.heuristics.get_value("num_samples", 10)
        )
        return Statistics.compute(
            samples, error_type=config.metrics.heuristics.get_value("error_type", "std")
        )
