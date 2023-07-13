from __future__ import annotations

import logging
import time
import warnings
from collections import defaultdict
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
from graphinf.data import DataModel
from graphinf.utility import EdgeCollector
from graphinf.utility import seed as gi_seed
from midynet.config import Config, DataModelFactory, GraphFactory, OptionError
from netrd import reconstruction as _reconstruction
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier

if "warnings" in dir(np):
    np.warnings.filterwarnings("ignore")
if "testing" in dir(np):
    np.testing.suppress_warnings()


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


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
    return (weights > threshold).astype("int")


class ProbabilityCalibrator:
    def __init__(self):
        self._trace = None

    @property
    def intercept(self) -> np.ndarray:
        if self._trace is None:
            raise RuntimeError(
                "The model has not been infered yet, cannot access `intercept`."
            )
        return self._trace.get_values("intercept")

    @property
    def coeff(self) -> np.ndarray:
        if self._trace is None:
            raise RuntimeError(
                "The model has not been infered yet, cannot access `coeff`."
            )
        return self._trace.get_values("coeff")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        steps: int = 1000,
        intercept_prior: Optional[Tuple[float, float]] = None,
        coeff_prior: Optional[Tuple[float, float]] = None,
    ) -> None:
        import pymc3

        a_mu, a_sd = (
            intercept_prior if isinstance(coeff_prior, tuple) else (0, 1)
        )
        b_mu, b_sd = (
            intercept_prior if isinstance(intercept_prior, tuple) else (0, 1)
        )
        logging.getLogger("pymc3").setLevel("ERROR")
        with pymc3.Model() as model:
            coeff = pymc3.Normal("coeff", mu=a_mu, sd=a_sd)
            intercept = pymc3.Normal("intercept", mu=b_mu, sd=b_sd)
            probs = pymc3.invlogit(intercept + coeff * X)
            likelihood = pymc3.Bernoulli("likelihood", p=probs, observed=y)
            self._trace = pymc3.sample(
                steps, chains=1, progressbar=False, return_inferencedata=False
            )

    def transform(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(
            self.intercept.reshape(-1, 1)
            + self.coeff.reshape(-1, 1) * X.reshape(1, -1)
        ).mean()

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, **kwargs
    ) -> np.ndarray:
        self.fit(X, y, **kwargs)
        return np.log(self.transform(X))


class Reconstructor:
    def __init__(self):
        self.clear()

    @property
    def pred(self):
        return self.__results__["pred"]

    def fit(self, obs, **kwargs):
        raise NotImplementedError()

    def compare(self, true_graph, measures=None, **kwargs):
        if len(self.__results__) == 0:
            raise ValueError("`results` must not be empty.")

        if isinstance(measures, str):
            measures = measures.split(", ")
        measures = ["roc"] if measures is None else measures
        true = np.array(true_graph.get_adjacency_matrix(True))
        np.fill_diagonal(true, 0)

        true[true > 1] = 1
        out = dict()
        for m in measures:
            if hasattr(self, "collect_" + m):
                m_dict = getattr(self, "collect_" + m)(true, self.pred)
                self.__results__.update(m_dict)
                out.update(m_dict)
            else:
                warnings.warn(
                    f"no collector named `{m}` has been found, proceeding anyway.",
                    RuntimeWarning,
                )
        return out

    def clear(self):
        self.__results__ = {}

    def normalize_weights(self, weights):
        if weights.min() == weights.max():
            return weights
        weights = (weights - weights.min()) / (weights.max() - weights.min())
        return weights

    def collect_accuracy(self, true, pred, **kwargs):
        pred_adj = threshold_weights(pred, true.sum())

        return dict(accuracy=accuracy_score(true, pred_adj))

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
            return dict(fpr=[0], tpr=[0], auc=0, thresholds=[0])
        auc = roc_auc_score(true, pred)

        return dict(fpr=fpr, tpr=tpr, auc=auc, thresholds=thresholds)

    def collect_error_prob(self, true, pred, **kwargs):
        if "prob" in self.__results__:
            logits = np.log(self.__results__["pred"])
        else:
            calibrator = ProbabilityCalibrator()
            logits = calibrator.fit_transform(pred, true, **kwargs)
        logp = np.sum(logits[true == 1]) + np.sum(
            np.log(1 - np.exp(logits[true == 0]))
        )
        error_prob = 1 - np.exp(logp)
        return dict(error_prob=error_prob)

    def collect_posterior_similarity(self, true, pred, **kwargs):
        true = (true > 0).astype("int")
        if "prob" in self.__results__:
            logits = np.log(self.__results__["pred"])
        else:
            calibrator = ProbabilityCalibrator()
            logits = calibrator.fit_transform(pred, true, **kwargs)
        pred = sigmoid(logits)
        posterior_similarity = 1 - np.sum(np.abs(true - pred)) / np.sum(
            np.abs(true + pred)
        )
        return dict(posterior_similarity=posterior_similarity)


class WeightbasedReconstructor(Reconstructor):
    def __init__(self, model, nanfill=None):
        self.model = model
        self.nanfill = 0 if nanfill is None else nanfill
        super().__init__()

    @ignore_warnings
    def fit(self, timeseries, **kwargs):
        self.clear()
        self.model.fit(timeseries, **kwargs)
        weights = self.model.results["weights_matrix"]
        weights[np.isnan(weights)] = self.nanfill
        np.fill_diagonal(weights, 0)
        self.__results__["pred"] = weights


class GraphbasedReconstructor(Reconstructor):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def fit(self, timeseries, **kwargs):
        self.clear()
        self.model.fit(timeseries, **kwargs)
        weights = nx.to_numpy_array(self.model.results["graph"])
        self.__results__["pred"] = weights


class BayesianReconstructor(Reconstructor):
    def __init__(self, config):
        self.graph = GraphFactory.build(config.prior)
        self.model = DataModelFactory.build(config.data_model)
        self.model.set_graph_prior(self.graph)
        self.__results__ = dict()

    def fit(self, g0=None, **kwargs):
        if kwargs.get("start_from_original", False):
            self.model.sample_prior()
        collector = EdgeCollector()
        if g0 is not None:
            collector.update(g0)
        collector.update(self.model.get_graph())
        for _ in range(kwargs.get("n_sweeps", 100)):
            self.model.metropolis_sweep(
                kwargs.get("n_step_per_nodes", 10) * self.model.get_size()
            )
            collector.update(self.model.get_graph())
        self.__results__["prob"] = {
            e: 1 - collector.mle(e, 0)
            for e in collector.multiplicities.keys()
        }
        self.__results__["pred"] = np.zeros(2 * (self.model.get_size(),))
        for e, p in self.__results__["prob"].items():
            self.__results__["pred"][e] = p
            if e[0] != e[1]:
                self.__results__["pred"][e[1], e[0]] = p


def get_reconstructor(config):
    reconstructors = {
        "correlation": lambda config: WeightbasedReconstructor(
            _reconstruction.CorrelationMatrix()
        ),
        "granger_causality": lambda config: WeightbasedReconstructor(
            _reconstruction.GrangerCausality()
        ),
        "transfer_entropy": lambda config: WeightbasedReconstructor(
            _reconstruction.NaiveTransferEntropy()
        ),
        "graphical_lasso": lambda config: WeightbasedReconstructor(
            _reconstruction.GraphicalLasso()
        ),
        "mutual_information": lambda config: WeightbasedReconstructor(
            _reconstruction.MutualInformationMatrix()
        ),
        "partial_correlation": lambda config: WeightbasedReconstructor(
            _reconstruction.PartialCorrelationMatrix()
        ),
        "correlation_spanning_tree": lambda config: GraphbasedReconstructor(
            _reconstruction.CorrelationSpanningTree()
        ),
    }

    if config.reconstructor in reconstructors:
        return reconstructors[config.reconstructor](config)
    else:
        raise OptionError(
            actual=config.reconstructor, expected=reconstructors.keys()
        )


class Predictor:
    def __init__(self):
        self.clear()

    def fit(self, inputs, targets, **kwargs):
        raise NotImplementedError()

    def clear(self):
        self.__results__ = {}

    def eval(self, targets, preds, measures=None, **kwargs):
        if isinstance(measures, str):
            measures = measures.split(", ")

        measures = ["absolute_error"] if len(measures) == 0 else measures
        out = dict()
        for m in measures:
            if hasattr(self, "collect_" + m):
                m_dict = getattr(self, "collect_" + m)(targets, preds)
                out.update(m_dict)
            else:
                warnings.warn(
                    f"no collector named `{m}` has been found, proceeding anyway.",
                    RuntimeWarning,
                )
        self.__results__.update(out)
        return out

    def collect_absolute_error(self, targets, preds, **kwargs):
        return dict(absolute_error=np.abs(targets - preds).mean())

    def collect_mean_square_error(self, targets, preds, **kwargs):
        return dict(mean_square_error=((targets - preds) ** 2).mean())


def prepare_training_data(config: Config, n_train_samples: int = 100):
    inputs = []
    targets = []

    prior = GraphFactory.build(config.prior)
    model = DataModelFactory.build(config.data_model)
    model.set_graph_prior(prior)

    for _ in range(n_train_samples):
        model.sample()
        inputs.append(np.array(model.get_past_states()).T)
        targets.append(np.array(model.get_future_states()).T)
    return np.concatenate(inputs, axis=0), np.concatenate(targets, axis=0)


class AverageProbabilityPredictor(Predictor):
    def __init__(self, config):
        self.config = config
        super().__init__()
        self.prior = GraphFactory.build(config.prior)
        self.model = DataModelFactory.build(config.data_model)
        self.model.set_graph_prior(self.prior)

    def fit(self, inputs, targets, **kwargs):
        self.avg_probs = []
        self.model.set_state(inputs, targets)
        for _ in range(kwargs.get("n_train_samples", 100)):
            self.model.sample_prior()
            self.avg_probs.append(
                np.array(self.model.get_transition_matrix(1)).T
            )
        self.avg_probs = np.array(self.avg_probs).mean(axis=0)
        self.__results__["pred"] = self.avg_probs

    def predict(self, inputs):
        return self.avg_probs


class MLEPredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.mle = {}

    def fit(self, inputs, targets, **kwargs):
        self.mle = defaultdict(list)
        for x, y in zip(inputs, targets):
            self.mle[tuple(x.tolist())].append(y)
        self.mle = {x: np.array(y).mean(0) for x, y in self.mle.items()}
        self.__results__["pred"] = self.mle
        return self.mle

    def predict(self, inputs):
        return np.array(
            [
                self.mle[x] if x in self.mle else 0.0
                for x in map(lambda x: tuple(x.tolist()), inputs)
            ]
        )


class LogisticPredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.regressor = None

    def fit(self, inputs, targets, **kwargs):
        self.regressor = [
            LogisticRegression() for _ in range(targets.shape[1])
        ]
        for i, reg in enumerate(self.regressor):
            reg.fit(inputs, targets[:, i])

    def predict(self, inputs):
        return np.array(
            [reg.predict_proba(inputs)[..., -1] for reg in self.regressor]
        ).T


class MLPPredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.mlp = None

    def fit(self, inputs, targets, **kwargs):
        self.mlp = MLPClassifier()
        self.mlp.fit(inputs, targets)

    def predict(self, inputs):
        return self.mlp.predict_proba(inputs)


def get_predictor(config):
    predictors = {
        "mle": lambda config: MLEPredictor(),
        "logistic": lambda config: LogisticPredictor(),
        "mlp": lambda config: MLPPredictor(),
    }

    if config.predictor in predictors:
        return predictors[config.predictor](config)
    else:
        raise OptionError(actual=config.predictor, expected=predictors.keys())
