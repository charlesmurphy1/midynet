from __future__ import annotations
import networkx as nx
import numpy as np
import time
import warnings

from collections import defaultdict
from graphinf.utility import seed as gi_seed
from midynet.config import Config, OptionError, GraphFactory, DataModelFactory
from .metrics import Metrics
from .multiprocess import Expectation
from midynet.statistics import Statistics
from dataclasses import dataclass, field
from netrd import reconstruction as _reconstruction
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from midynet.aggregator import Aggregator
from scipy.optimize import minimize_scalar

def ignore_warnings(func):
    def wrapper(*args, **kwargs):
        np.seterr(invalid="ignore")
        warnings.filterwarnings("ignore")

        value = func(*args, **kwargs)

        np.seterr(invalid="ignore")
        warnings.filterwarnings("ignore")

        return value

    return wrapper

def threshold_weights(weights, edge_count):
    f_to_solve = lambda t: np.abs(np.sum(weights > t) - edge_count)
    threshold = minimize_scalar(f_to_solve)["x"]
    # threshold = bisect(f_to_solve, weights.min(), weights.max())
    return (weights > threshold).astype("int")

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
                self.__results__[m] = getattr(self, "collect_" + m)(
                    true, self.pred
                )
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
    
    def collect_accuracy(self, true, pred, **kwargs):
        pred_adj = threshold_weights(pred, true.sum())
        return accuracy_score(true, pred_adj)


    def collect_confusion_matrix(self, true, pred, **kwargs):
        threshold = kwargs.get("threshold", norm_pred.mean())
        norm_pred = self.normalize_weights(pred).reshape(-1)
        true = true.reshape(-1)
        cm = confusion_matrix(
            true, (norm_pred > threshold).astype("float").reshape(-1)
        )
        tn, fp, fn, tp = cm.ravel()

        return dict(threshold=threshold, tn=tn, fp=fp, fn=fn, tp=tp)

    def collect_roc(self, true, pred, **kwargs):
        pred[np.isnan(pred)] = -10
        pred = self.normalize_weights(pred).reshape(-1).astype("float")
        true = true.reshape(-1).astype("int")
        fpr, tpr, thresholds = roc_curve(true, pred)
        if len(np.unique(true)) == 1:
            return dict(fpr=0, tpr=0, auc=0, thresholds=0)
        auc = roc_auc_score(true, pred)

        return dict(fpr=fpr, tpr=tpr, auc=auc, thresholds=thresholds)


class WeightbasedReconstructionHeuristicsMethod(
    ReconstructionHeuristicsMethod
):
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


class GraphbasedReconstructionHeuristicsMethod(
    ReconstructionHeuristicsMethod
):
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
        raise OptionError(
            actual=config.method, expected=reconstructors.keys()
        )


class ReconstructionHeuristics(Expectation):
    def __init__(self, config: Config, **kwargs):
        self.params = config.dict
        super().__init__(**kwargs)

    def func(self, seed: int) -> float:
        gi_seed(seed)
        config = Config.from_dict(self.params)
        graph_model = GraphFactory.build(config.prior)
        data_model = DataModelFactory.build(config.data_model)
        data_model.set_graph_prior(graph_model)

        data_model.sample()
        timeseries = np.array(data_model.get_past_states()).T
        heuristics = get_heuristics_reconstructor(config.metrics)
        heuristics.fit(timeseries)
        heuristics.compare(graph_model.get_state(), measures=["roc", "accuracy"])
        return dict(auc=heuristics.__results__["roc"]["auc"], accuracy=heuristics.__results__["accuracy"])


class ReconstructionHeuristicsMetrics(Metrics):
    shortname = "reconheuristics"
    keys = ["auc", "accuracy"]

    def eval(self, config: Config):
        heuristics = ReconstructionHeuristics(
            config=config,
            num_procs=config.get("num_procs", 1),
            seed=config.get("seed", int(time.time())),
        )
        samples = heuristics.compute(config.metrics.get("num_samples", 10))
        sample_dict = defaultdict(list)
        for s in samples:
            for k, v in s.items():
                sample_dict[k].append(v)

        if config.metrics.reduction == "identity":
            return sample_dict
        stats = {}
        for k, v in sample_dict.items():
            stats[k] = Statistics.from_samples(
                v,
                reduction=config.metrics.get("reduction", "normal"),
                name=k,
            )

        out = dict()
        for k, s in stats.items():
            for sk, sv in s.__data__.items():
                out[k + "_" + sk] = [sv]
        return out
